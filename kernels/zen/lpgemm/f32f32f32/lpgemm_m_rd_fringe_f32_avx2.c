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

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_2x16_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    __m256  ymm0,  ymm1,  ymm2,  ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 16; jj += 4 )    // LOOP_2x16J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

                ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

                ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

                ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

                ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                a_temp += 8;
                b_temp += 8;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
        {
            ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += 8;
            b_temp += 8;
        }

        if ( k_left1 )
        {
            const __m256i mask = masks[k_left1];

            ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
            ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

            ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += k_left1;
            b_temp += k_left1;
        }

        // ACCUMULATE
        ymm0 = _mm256_hadd_ps(ymm4, ymm7);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm10, ymm13);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm4 = _mm_hadd_ps(xmm0, xmm2);

        ymm0 = _mm256_hadd_ps(ymm5, ymm8);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm11, ymm14);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm5 = _mm_hadd_ps(xmm0, xmm2);

        // ALPHA SCALE
        xmm0 = _mm_broadcast_ss(&alpha);
        xmm4 = _mm_mul_ps(xmm4, xmm0);
        xmm5 = _mm_mul_ps(xmm5, xmm0);

        if ( beta != 0 )
        {
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);

            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x16F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {

                if ( post_ops_list_temp->stor_type == BF16 )
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_2x16F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            // c[1,0-3]
            xmm5 = _mm_max_ps( xmm5, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_2x16F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_2x16F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            // c[1,0-3]
            GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_2x16F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_2x16F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            // c[1,0-3]
            CLIP_F32S_SSE(xmm5, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_2x16F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_j + ( 0 * 4) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                    }
                    else
                    {
                        zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
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
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 1 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_2x16F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_2x16F:
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

POST_OPS_SWISH_2x16F:
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

POST_OPS_TANH_2x16F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            // c[1,0-3]
            TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_2x16F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            // c[1,0-3]
            SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_2x16F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
        }

        post_ops_attr.post_op_c_j += 4;

        cbuf = cbuf + 3*rs_c0;
        abuf = abuf + 3*rs_a0;
    }   // END LOOP_2x16J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_1x16_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    __m256  ymm0,  ymm1,  ymm2,  ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 16; jj += 4 )    // LOOP_1x16J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

                ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

                ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

                ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

                ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                a_temp += 8;
                b_temp += 8;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
        {
            ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += 8;
            b_temp += 8;
        }

        if ( k_left1 )
        {
            const __m256i mask = masks[k_left1];

            ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
            ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

            ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += k_left1;
            b_temp += k_left1;
        }

        // ACCUMULATE
        ymm0 = _mm256_hadd_ps(ymm4, ymm7);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm10, ymm13);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm4 = _mm_hadd_ps(xmm0, xmm2);

        ymm0 = _mm256_hadd_ps(ymm5, ymm8);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm11, ymm14);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm5 = _mm_hadd_ps(xmm0, xmm2);

        // ALPHA SCALE
        xmm0 = _mm_broadcast_ss(&alpha);
        xmm4 = _mm_mul_ps(xmm4, xmm0);
        xmm5 = _mm_mul_ps(xmm5, xmm0);

        if ( beta != 0 )
        {
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);

            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x16F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {

                if ( post_ops_list_temp->stor_type == BF16 )
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_1x16F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_1x16F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_1x16F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_1x16F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_1x16F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_1x16F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
                }
                else
                {
                    zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                }
            }

            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_j + ( 0 * 4) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                    }
                    else
                    {
                        zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                                                        post_ops_attr.post_op_c_j + ( 0 * 4 ) );
                    }
                }

                //c[0, 0-3]
                F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_1x16F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_1x16F:
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

POST_OPS_SWISH_1x16F:
        {
            xmm0 =
                _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_TANH_1x16F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_1x16F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_1x16F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
        }

        post_ops_attr.post_op_c_j += 4;

        cbuf = cbuf + 3*rs_c0;
        abuf = abuf + 3*rs_a0;
    }   // END LOOP_1x16J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_2x8_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    __m256  ymm0,  ymm1,  ymm2,  ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 8; jj += 4 )    // LOOP_2x8J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

                ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

                ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

                ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

                ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                a_temp += 8;
                b_temp += 8;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
        {
            ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += 8;
            b_temp += 8;
        }

        if ( k_left1 )
        {
            const __m256i mask = masks[k_left1];

            ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
            ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

            ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += k_left1;
            b_temp += k_left1;
        }

        // ACCUMULATE
        ymm0 = _mm256_hadd_ps(ymm4, ymm7);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm10, ymm13);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm4 = _mm_hadd_ps(xmm0, xmm2);

        ymm0 = _mm256_hadd_ps(ymm5, ymm8);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm11, ymm14);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm5 = _mm_hadd_ps(xmm0, xmm2);

        // ALPHA SCALE
        xmm0 = _mm_broadcast_ss(&alpha);
        xmm4 = _mm_mul_ps(xmm4, xmm0);
        xmm5 = _mm_mul_ps(xmm5, xmm0);

        if ( beta != 0 )
        {
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);

            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x8F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {

                if ( post_ops_list_temp->stor_type == BF16 )
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_2x8F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            // c[1,0-3]
            xmm5 = _mm_max_ps( xmm5, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_2x8F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_2x8F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            // c[1,0-3]
            GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_2x8F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_2x8F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            // c[1,0-3]
            CLIP_F32S_SSE(xmm5, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_2x8F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_j + ( 0 * 4) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                    }
                    else
                    {
                        zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
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
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 1 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_2x8F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_2x8F:
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

POST_OPS_SWISH_2x8F:
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

POST_OPS_TANH_2x8F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            // c[1,0-3]
            TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_2x8F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            // c[1,0-3]
            SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_2x8F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
        }

        post_ops_attr.post_op_c_j += 4;

        cbuf = cbuf + 3*rs_c0;
        abuf = abuf + 3*rs_a0;
    }   // END LOOP_2x8J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_1x8_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    __m256  ymm0,  ymm1,  ymm2,  ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 8; jj += 4 )    // LOOP_1x8J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

                ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

                ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

                ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

                ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                a_temp += 8;
                b_temp += 8;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
        {
            ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += 8;
            b_temp += 8;
        }

        if ( k_left1 )
        {
            const __m256i mask = masks[k_left1];

            ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
            ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

            ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += k_left1;
            b_temp += k_left1;
        }

        // ACCUMULATE
        ymm0 = _mm256_hadd_ps(ymm4, ymm7);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm10, ymm13);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm4 = _mm_hadd_ps(xmm0, xmm2);

        ymm0 = _mm256_hadd_ps(ymm5, ymm8);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm11, ymm14);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm5 = _mm_hadd_ps(xmm0, xmm2);

        // ALPHA SCALE
        xmm0 = _mm_broadcast_ss(&alpha);
        xmm4 = _mm_mul_ps(xmm4, xmm0);
        xmm5 = _mm_mul_ps(xmm5, xmm0);

        if ( beta != 0 )
        {
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);

            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x8F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {

                if ( post_ops_list_temp->stor_type == BF16 )
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_1x8F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_1x8F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_1x8F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_1x8F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_1x8F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_1x8F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
                }
                else
                {
                    zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                }
            }

            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_j + ( 0 * 4) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                    }
                    else
                    {
                        zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                                                        post_ops_attr.post_op_c_j + ( 0 * 4 ) );
                    }
                }

                //c[0, 0-3]
                F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_1x8F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_1x8F:
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

POST_OPS_SWISH_1x8F:
        {
            xmm0 =
                _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_TANH_1x8F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_1x8F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_1x8F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
        }

        post_ops_attr.post_op_c_j += 4;

        cbuf = cbuf + 3*rs_c0;
        abuf = abuf + 3*rs_a0;
    }   // END LOOP_1x8J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_2x4_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    __m256  ymm0,  ymm1,  ymm2, ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 4; jj += 4 )    // LOOP_2x4J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

                ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

                ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

                ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

                ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                a_temp += 8;
                b_temp += 8;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
        {
            ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += 8;
            b_temp += 8;
        }

        if ( k_left1 )
        {
            const __m256i mask = masks[k_left1];

            ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
            ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

            ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += k_left1;
            b_temp += k_left1;
        }

        // ACCUMULATE
        ymm0 = _mm256_hadd_ps(ymm4, ymm7);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm10, ymm13);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm4 = _mm_hadd_ps(xmm0, xmm2);

        ymm0 = _mm256_hadd_ps(ymm5, ymm8);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm11, ymm14);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm5 = _mm_hadd_ps(xmm0, xmm2);

        // ALPHA SCALE
        xmm0 = _mm_broadcast_ss(&alpha);
        xmm4 = _mm_mul_ps(xmm4, xmm0);
        xmm5 = _mm_mul_ps(xmm5, xmm0);

        if ( beta != 0 )
        {
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);

            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x4F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {

                if ( post_ops_list_temp->stor_type == BF16 )
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
                if ( post_ops_list_temp->stor_type == BF16 )
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
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_j + ( 0 * 4) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                    }
                    else
                    {
                        zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
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
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 1 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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
            if ( is_bf16 == TRUE )
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
        }

        post_ops_attr.post_op_c_j += 4;

        cbuf = cbuf + 3*rs_c0;
        abuf = abuf + 3*rs_a0;
    }   // END LOOP_2x8J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_1x4_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    __m256  ymm0,  ymm1,  ymm2,  ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 4; jj += 4 )    // LOOP_1x4J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

                ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

                ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

                ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

                ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                a_temp += 8;
                b_temp += 8;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
        {
            ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_loadu_ps(b_temp + 2*cs_b0);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_loadu_ps(b_temp + 3*cs_b0);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += 8;
            b_temp += 8;
        }

        if ( k_left1 )
        {
            const __m256i mask = masks[k_left1];

            ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
            ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

            ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            ymm3  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
            ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

            ymm3  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

            a_temp += k_left1;
            b_temp += k_left1;
        }

        // ACCUMULATE
        ymm0 = _mm256_hadd_ps(ymm4, ymm7);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm10, ymm13);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm4 = _mm_hadd_ps(xmm0, xmm2);

        ymm0 = _mm256_hadd_ps(ymm5, ymm8);
        xmm1 = _mm256_extractf128_ps(ymm0, 1);
        xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
        ymm2 = _mm256_hadd_ps(ymm11, ymm14);
        xmm1 = _mm256_extractf128_ps(ymm2, 1);
        xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);

        xmm5 = _mm_hadd_ps(xmm0, xmm2);

        // ALPHA SCALE
        xmm0 = _mm_broadcast_ss(&alpha);
        xmm4 = _mm_mul_ps(xmm4, xmm0);
        xmm5 = _mm_mul_ps(xmm5, xmm0);

        if ( beta != 0 )
        {
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);

            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x4F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {

                if ( post_ops_list_temp->stor_type == BF16 )
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
                if ( post_ops_list_temp->stor_type == BF16 )
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
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
                }
                else
                {
                    zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                }
            }

            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_j + ( 0 * 4) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                    }
                    else
                    {
                        zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                                                        post_ops_attr.post_op_c_j + ( 0 * 4 ) );
                    }
                }

                //c[0, 0-3]
                F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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
            if ( is_bf16 == TRUE )
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
            _mm_storeu_ps(c_temp, xmm4);
        }

        post_ops_attr.post_op_c_j += 4;

        cbuf = cbuf + 3*rs_c0;
        abuf = abuf + 3*rs_a0;
    }   // END LOOP_1x4J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_2x2_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    __m128i m_mask = _mm_set_epi32(0,  0, -1, -1);

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;

    __m256 ymm0, ymm1, ymm3, ymm4, ymm5, ymm7, ymm8;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    lpgemm_post_op* post_ops_list_temp = post_ops_list;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    float* c_temp = cbuf;
    float* a_temp = abuf;
    float* b_temp = bbuf;

    // zero out all ymm registers
    ZERO_ACC_YMM_4_REG(ymm0, ymm1, ymm3, ymm4)
    ZERO_ACC_YMM_3_REG(ymm5, ymm7, ymm8)

    // zero out all xmm registers
    ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
    ZERO_ACC_XMM_2_REG(xmm4, xmm5)

    for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
    {
        for ( dim_t unroll = 0; unroll < 4; ++unroll )
        {
            ymm0 = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1 = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3 = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            ymm3 = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
            ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

            a_temp += 8;
            b_temp += 8;
        }
    }

    for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
    {
        ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
        ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

        ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
        ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
        ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

        ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
        ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
        ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

        a_temp += 8;
        b_temp += 8;
    }

    if ( k_left1 )
    {
        const __m256i mask = masks[k_left1];

        ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
        ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

        ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
        ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

        ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
        ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
        ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);

        a_temp += k_left1;
        b_temp += k_left1;
    }

    // ACCUMULATE
    ymm0 = _mm256_hadd_ps(ymm4, ymm7);
    xmm1 = _mm256_extractf128_ps(ymm0, 1);
    xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);

    xmm4 = _mm_hadd_ps(xmm0, xmm0);

    ymm0 = _mm256_hadd_ps(ymm5, ymm8);
    xmm1 = _mm256_extractf128_ps(ymm0, 1);
    xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);

    xmm5 = _mm_hadd_ps(xmm0, xmm0);

    // ALPHA SCALE
    xmm0 = _mm_broadcast_ss(&alpha);
    xmm4 = _mm_mul_ps(xmm4, xmm0);
    xmm5 = _mm_mul_ps(xmm5, xmm0);

    if ( beta != 0 )
    {
        xmm3 = _mm_broadcast_ss(&beta);

        xmm0 = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
        xmm4 = _mm_fmadd_ps(xmm3, xmm0, xmm4);

        xmm0 = _mm_maskload_ps(c_temp + 1*rs_c0, m_mask);
        xmm5 = _mm_fmadd_ps(xmm3, xmm0, xmm5);
    }

    // Post Ops
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x2F:
    {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {

            if ( post_ops_list_temp->stor_type == BF16 )
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
            if ( post_ops_list_temp->stor_type == BF16 )
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
        xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
        {
            if ( is_bf16 == TRUE )
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

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_j + ( 0 * 4) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                }
                else
                {
                    zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
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
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 0 ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 1 ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
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
            scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 0 ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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
        if ( is_bf16 == TRUE )
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

POST_OPS_SWISH_2x2F:
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

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_2x2F_DISABLE:
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
        _mm_maskstore_ps(c_temp, m_mask, xmm4);
        c_temp += rs_c;
        _mm_maskstore_ps(c_temp, m_mask, xmm5);
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_1x2_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    __m128i m_mask = _mm_set_epi32(0,  0, -1, -1);

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;

    __m256 ymm0, ymm3, ymm4, ymm7;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

    lpgemm_post_op* post_ops_list_temp = post_ops_list;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    float* c_temp = cbuf;
    float* a_temp = abuf;
    float* b_temp = bbuf;

    // zero out all ymm registers
    ZERO_ACC_YMM_4_REG(ymm0, ymm3, ymm4, ymm7)

    // zero out all xmm registers
    ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
    xmm4 = _mm_setzero_ps();

    for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
    {
        for ( dim_t unroll = 0; unroll < 4; ++unroll )
        {
            ymm0 = _mm256_loadu_ps(a_temp + 0*rs_a0);

            ymm3 = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

            ymm3 = _mm256_loadu_ps(b_temp + 1*cs_b0);
            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);

            a_temp += 8;
            b_temp += 8;
        }
    }

    for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
    {
        ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);

        ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
        ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

        ymm3  = _mm256_loadu_ps(b_temp + 1*cs_b0);
        ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);

        a_temp += 8;
        b_temp += 8;
    }

    if ( k_left1 )
    {
        const __m256i mask = masks[k_left1];

        ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);

        ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

        ymm3 = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
        ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);

        a_temp += k_left1;
        b_temp += k_left1;
    }

    // ACCUMULATE
    ymm0 = _mm256_hadd_ps(ymm4, ymm7);
    xmm1 = _mm256_extractf128_ps(ymm0, 1);
    xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);

    xmm4 = _mm_hadd_ps(xmm0, xmm0);

    // ALPHA SCALE
    xmm0 = _mm_broadcast_ss(&alpha);
    xmm4 = _mm_mul_ps(xmm4, xmm0);

    if ( beta != 0 )
    {
        xmm3 = _mm_broadcast_ss(&beta);

        xmm0 = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
        xmm4 = _mm_fmadd_ps(xmm3, xmm0, xmm4);
    }

    // Post Ops
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x2F:
    {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
            if ( post_ops_list_temp->stor_type == BF16 )
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
            if ( post_ops_list_temp->stor_type == BF16 )
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
        xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
        {
            if ( is_bf16 == TRUE )
            {
                BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
            }
            else
            {
                zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            }
        }

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_j + ( 0 * 4) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                }
                else
                {
                    zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                                                    post_ops_attr.post_op_c_j + ( 0 * 4 ) );
                }
            }

            //c[0, 0-3]
            F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
        }
        else
        {
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 0 ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
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
            scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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
        if ( is_bf16 == TRUE )
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

POST_OPS_SWISH_1x2F:
    {
        xmm0 =
            _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
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

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_1x2F_DISABLE:
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
        _mm_maskstore_ps(c_temp, m_mask, xmm4);
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_2x1_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    __m128i m_mask = _mm_set_epi32(0, 0, 0, -1);

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;

    __m256 ymm0, ymm1, ymm3, ymm4, ymm5;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    lpgemm_post_op* post_ops_list_temp = post_ops_list;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    float* c_temp = cbuf;
    float* a_temp = abuf;
    float* b_temp = bbuf;

    // zero out all ymm registers
    ZERO_ACC_YMM_4_REG(ymm0, ymm1, ymm3, ymm4)
    ymm5 = _mm256_setzero_ps();

    // zero out all xmm registers
    ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
    ZERO_ACC_XMM_2_REG(xmm4, xmm5)

    for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
    {
        for ( dim_t unroll = 0; unroll < 4; ++unroll )
        {
            ymm0 = _mm256_loadu_ps(a_temp + 0*rs_a0);
            ymm1 = _mm256_loadu_ps(a_temp + 1*rs_a0);

            ymm3 = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
            ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

            a_temp += 8;
            b_temp += 8;
        }
    }

    for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
    {
        ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);
        ymm1  = _mm256_loadu_ps(a_temp + 1*rs_a0);

        ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
        ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
        ymm5  = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

        a_temp += 8;
        b_temp += 8;
    }

    if ( k_left1 )
    {
        const __m256i mask = masks[k_left1];

        ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
        ymm1 = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);

        ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
        ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

        a_temp += k_left1;
        b_temp += k_left1;
    }

    // ACCUMULATE
    ymm0 = _mm256_hadd_ps(ymm4, ymm4);
    ymm1 = _mm256_hadd_ps(ymm0, ymm0);
    xmm0 = _mm256_extractf128_ps(ymm1, 1);
    xmm4 = _mm_add_ps(_mm256_castps256_ps128(ymm1), xmm0);

    ymm0 = _mm256_hadd_ps(ymm5, ymm5);
    ymm1 = _mm256_hadd_ps(ymm0, ymm0);
    xmm0 = _mm256_extractf128_ps(ymm1, 1);
    xmm5 = _mm_add_ps(_mm256_castps256_ps128(ymm1), xmm0);

    // ALPHA SCALE
    xmm0 = _mm_broadcast_ss(&alpha);
    xmm4 = _mm_mul_ps(xmm4, xmm0);
    xmm5 = _mm_mul_ps(xmm5, xmm0);

    if ( beta != 0 )
    {
        xmm3 = _mm_broadcast_ss(&beta);

        xmm0 = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
        xmm4 = _mm_fmadd_ps(xmm3, xmm0, xmm4);

        xmm0 = _mm_maskload_ps(c_temp + 1*rs_c0, m_mask);
        xmm5 = _mm_fmadd_ps(xmm3, xmm0, xmm5);
    }

    // Post Ops
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x1F:
    {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {

            if ( post_ops_list_temp->stor_type == BF16 )
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
            if ( post_ops_list_temp->stor_type == BF16 )
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
        xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
        {
            if ( is_bf16 == TRUE )
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

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_j + ( 0 * 4) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                }
                else
                {
                    zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
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
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 0 ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 1 ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
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
            scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 0 ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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
        if ( is_bf16 == TRUE )
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

POST_OPS_SWISH_2x1F:
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

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_2x1F_DISABLE:
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
        _mm_maskstore_ps(c_temp, m_mask, xmm4);
        c_temp += rs_c;
        _mm_maskstore_ps(c_temp, m_mask, xmm5);
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_1x1_rd)
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

    __m256i masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)     // 7 elements
    };

    __m128i m_mask = _mm_set_epi32(0, 0, 0, -1);

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;

    __m256 ymm0, ymm1, ymm3, ymm4;
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

    lpgemm_post_op* post_ops_list_temp = post_ops_list;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    float* c_temp = cbuf;
    float* a_temp = abuf;
    float* b_temp = bbuf;

    // zero out all ymm registers
    ZERO_ACC_YMM_4_REG(ymm0, ymm1, ymm4, ymm4)

    // zero out all xmm registers
    ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
    xmm4 = _mm_setzero_ps();

    for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
    {
        for ( dim_t unroll = 0; unroll < 4; ++unroll )
        {
            ymm0 = _mm256_loadu_ps(a_temp + 0*rs_a0);

            ymm3 = _mm256_loadu_ps(b_temp + 0*cs_b0);
            ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

            a_temp += 8;
            b_temp += 8;
        }
    }

    for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
    {
        ymm0  = _mm256_loadu_ps(a_temp + 0*rs_a0);

        ymm3  = _mm256_loadu_ps(b_temp + 0*cs_b0);
        ymm4  = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

        a_temp += 8;
        b_temp += 8;
    }

    if ( k_left1 )
    {
        const __m256i mask = masks[k_left1];

        ymm0 = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);

        ymm3 = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

        a_temp += k_left1;
        b_temp += k_left1;
    }

    // ACCUMULATE
    ymm0 = _mm256_hadd_ps(ymm4, ymm4);
    ymm1 = _mm256_hadd_ps(ymm0, ymm0);
    xmm0 = _mm256_extractf128_ps(ymm1, 1);
    xmm4 = _mm_add_ps(_mm256_castps256_ps128(ymm1), xmm0);

    // ALPHA SCALE
    xmm0 = _mm_broadcast_ss(&alpha);
    xmm4 = _mm_mul_ps(xmm4, xmm0);

    if ( beta != 0 )
    {
        xmm3 = _mm_broadcast_ss(&beta);

        xmm0 = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
        xmm4 = _mm_fmadd_ps(xmm3, xmm0, xmm4);
    }

    // Post Ops
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x1F:
    {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {

            if ( post_ops_list_temp->stor_type == BF16 )
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
            if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_1x1F:
    {
        xmm0 = _mm_setzero_ps();

        // c[0,0-3]
        xmm4 = _mm_max_ps( xmm4, xmm0 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_RELU_SCALE_1x1F:
    {
        xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
        {
            if ( is_bf16 == TRUE )
            {
                BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
            }
            else
            {
                zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            }
        }

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_j + ( 0 * 4) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0, 0)
                }
                else
                {
                    zero_point0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                                                    post_ops_attr.post_op_c_j + ( 0 * 4 ) );
                }
            }

            //c[0, 0-3]
            F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
        }
        else
        {
            if ( post_ops_list_temp->scale_factor_len > 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                            post_ops_attr.post_op_c_i + 0 ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
            {
                if ( is_bf16 == TRUE )
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
            scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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
        if ( is_bf16 == TRUE )
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

POST_OPS_SWISH_1x1F:
    {
        xmm0 =
            _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
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

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_1x1F_DISABLE:
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
        _mm_maskstore_ps(c_temp, m_mask, xmm4);
    }
}
#endif
