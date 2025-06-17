/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

  Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "../../../zen/lpgemm/f32f32f32/lpgemm_kernel_macros_f32_avx2.h"

#define LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, paddr, stride ) \
    ymm0 = _mm256_loadu_ps( paddr ); \
    ymm1 = _mm256_loadu_ps( paddr + stride ); \
    ymm2 = _mm256_loadu_ps( paddr + 2 * stride ); \
    ymm3 = _mm256_loadu_ps( paddr + 3 * stride );

#define LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, mask, paddr, stride ) \
    ymm0 = _mm256_maskload_ps( paddr, mask ); \
    ymm1 = _mm256_maskload_ps( paddr + stride, mask ); \
    ymm2 = _mm256_maskload_ps( paddr + 2 * stride, mask ); \
    ymm3 = _mm256_maskload_ps( paddr + 3 * stride, mask );

#define LPGEMV_N_KERNEL_4_FMA( ymm8, ymm9, ymm10, ymm11, ymm7, ymm0, ymm1, ymm2, ymm3 ) \
    ymm8 = _mm256_fmadd_ps( ymm0, ymm7, ymm8 ); \
    ymm9 = _mm256_fmadd_ps( ymm1, ymm7, ymm9 ); \
    ymm10 = _mm256_fmadd_ps( ymm2, ymm7, ymm10 ); \
    ymm11 = _mm256_fmadd_ps( ymm3, ymm7, ymm11 );

#define LPGEMV_YMM2XMM( ymm8, ymm9, ymm10, ymm11, ymm0, ymm1, ymm2, ymm3, xmm0 ) \
    ymm0 = _mm256_hadd_ps( ymm8, ymm9 ); \
    ymm1 = _mm256_hadd_ps( ymm10, ymm11 ); \
    ymm0 = _mm256_hadd_ps( ymm0, ymm1 ); \
    xmm0 = _mm_add_ps(_mm256_extractf128_ps(ymm0, 0), _mm256_extractf128_ps(ymm0,1));


LPGEMV_N_EQ1_KERN( float, float, float, f32f32f32of32_avx512_256 )
{
    static void *post_ops_labels[] =
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

     // Strides are updated based on matrix packing/reordering.
     const float *a_use = NULL;
     const float *b_use = NULL;
     float *c_use = NULL;

     lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

     __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
     __m256 ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
     __m256 ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23;
     __m256 ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31;

     __m128 xmm0, xmm1, xmm8, xmm9;

     __m256i masks[9] = {
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32( 0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
        _mm256_set_epi32( 0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
        _mm256_set_epi32( 0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
        _mm256_set_epi32( 0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
        _mm256_set_epi32( 0, -1, -1, -1, -1, -1, -1, -1),    // 7 elements
        _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1)     // 8 elements
    };

    for (dim_t mr = 0; mr < m0; mr += MR)
    {
        dim_t mr0 = bli_min((m0 - mr), MR);

        dim_t k_iter = k/8;
        dim_t k_rem = k % 8;

        __m256i store_mask1, store_mask2;
        if( mr0 >= 8 )
        {
            store_mask1 = masks[8];
            store_mask2 = masks[mr0 - 8 ];
        }
        else
        {
            store_mask1 = masks[mr0];
            store_mask2 = masks[0];

        }

        const __m256i k_rem_mask = masks[k_rem];

        /* zero the accumulator registers */
        ZERO_ACC_YMM_4_REG( ymm16, ymm17, ymm18, ymm19 );
        ZERO_ACC_YMM_4_REG( ymm20, ymm21, ymm22, ymm23 );
        ZERO_ACC_YMM_4_REG( ymm24, ymm25, ymm26, ymm27 );
        ZERO_ACC_YMM_4_REG( ymm28, ymm29, ymm30, ymm31 );
        ymm13 = _mm256_setzero_ps();
        ymm14 = _mm256_setzero_ps();

        //update pointers
        a_use = a + mr * rs_a;
        b_use = b;
        c_use = c + mr * rs_c;

        if( mr0 == MR )
        {
            for( dim_t k = 0; k < k_iter; k++ )
            {
                ymm15 = _mm256_loadu_ps( b_use );
                b_use += 8; // move b pointer to next 16 elements

                // Load 4x8 from row 0-3 of A
                LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm16, ymm17, ymm18, ymm19, ymm15, ymm0, ymm1, ymm2, ymm3 );

                // Load 4x8 from row 4-7 of A
                LPGEMV_N_KERNEL_4_LOADS( ymm4, ymm5, ymm6, ymm7, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm20, ymm21, ymm22, ymm23, ymm15, ymm4, ymm5, ymm6, ymm7 );

                // Load 4x8 from row 8-11 of A
                LPGEMV_N_KERNEL_4_LOADS( ymm8, ymm9, ymm10, ymm11, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm24, ymm25, ymm26, ymm27, ymm15, ymm8, ymm9, ymm10, ymm11 );

                LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a );
                a_use -= 12 * rs_a; // move back the pointer to the last 4x8 elements
                LPGEMV_N_KERNEL_4_FMA( ymm28, ymm29, ymm30, ymm31, ymm15, ymm0, ymm1, ymm2, ymm3 );

                a_use += 8;
            } // k-loop

            if( k_rem )
            {
                ymm15 = _mm256_maskload_ps( b_use, k_rem_mask );

                // Load 4x8 from row 0-3 of A
                LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, k_rem_mask, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm16, ymm17, ymm18, ymm19, ymm15, ymm0, ymm1, ymm2, ymm3 );

                // Load 4x8 from row 4-7 of A
                LPGEMV_N_KERNEL_4_MASKLOADS( ymm4, ymm5, ymm6, ymm7, k_rem_mask, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm20, ymm21, ymm22, ymm23, ymm15, ymm4, ymm5, ymm6, ymm7 );

                // Load 4x8 from row 8-11 of A
                LPGEMV_N_KERNEL_4_MASKLOADS( ymm8, ymm9, ymm10, ymm11, k_rem_mask, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm24, ymm25, ymm26, ymm27, ymm15, ymm8, ymm9, ymm10, ymm11 );

                LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, k_rem_mask, a_use, rs_a );
                a_use -= 12 * rs_a; // move back the pointer to the last 4x8 elements

                LPGEMV_N_KERNEL_4_FMA( ymm28, ymm29, ymm30, ymm31, ymm15, ymm0, ymm1, ymm2, ymm3 );
                a_use += 8;
            }

            // Add the registers horizontally to get one output
            LPGEMV_YMM2XMM( ymm16, ymm17, ymm18, ymm19, ymm0, ymm1, ymm2, ymm3, xmm0 );
            LPGEMV_YMM2XMM( ymm20, ymm21, ymm22, ymm23, ymm4, ymm5, ymm6, ymm7, xmm1 );
            LPGEMV_YMM2XMM( ymm24, ymm25, ymm26, ymm27, ymm8, ymm9, ymm10, ymm11, xmm8 );
            LPGEMV_YMM2XMM( ymm28, ymm29, ymm30, ymm31, ymm4, ymm5, ymm6, ymm7, xmm9 );

            // compose outputs into one ymm to perform post-ops
            ymm30 = _mm256_insertf128_ps( ymm30, xmm0, 0 );
            ymm30 = _mm256_insertf128_ps( ymm30, xmm1, 1 );
            ymm31 = _mm256_insertf128_ps( ymm31, xmm8, 0 );
            ymm31 = _mm256_insertf128_ps( ymm31, xmm9, 1 );
        }
        else
        {

            // Handle fringe cases when mr0 < MR
            const float *a_use_fringe =  a_use;
            dim_t mr0_use = mr0;
            dim_t regidx = 0;
            bool is_mr_8 = 0;

            if( mr0_use >= 8 )
            {
                for( dim_t k = 0; k < k_iter; k++ )
                {
                    ymm15 = _mm256_loadu_ps( b_use );
                    b_use += 8; // move b pointer to next 8 elements

                    // Load 4x8 from row 0-3 of A
                    LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a );
                    a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm16, ymm17, ymm18, ymm19, ymm15, ymm0, ymm1, ymm2, ymm3 );

                    // Load 4x8 from row 4-7 of A
                    LPGEMV_N_KERNEL_4_LOADS( ymm4, ymm5, ymm6, ymm7, a_use, rs_a );
                    a_use -= 4 * rs_a; // move a pointer to next 4x8 elements

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm20, ymm21, ymm22, ymm23, ymm15, ymm4, ymm5, ymm6, ymm7 );

                    a_use += 8;
                } // k-loop

                if( k_rem )
                {
                    ymm15 = _mm256_maskload_ps( b_use, k_rem_mask );

                    // Load 4x8 from row 0-3 of A
                    LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, k_rem_mask, a_use, rs_a );
                    a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm16, ymm17, ymm18, ymm19, ymm15, ymm0, ymm1, ymm2, ymm3 );

                    // Load 4x8 from row 4-7 of A
                    LPGEMV_N_KERNEL_4_MASKLOADS( ymm4, ymm5, ymm6, ymm7, k_rem_mask, a_use, rs_a );
                    a_use -= 4 * rs_a; // move back the pointer to the last 4x8 elements

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm20, ymm21, ymm22, ymm23, ymm15, ymm4, ymm5, ymm6, ymm7 );
                }

                //update pointers
                mr0_use -= 8;
                a_use = a_use_fringe + 8 * rs_a;
                a_use_fringe = a_use;
                b_use = b;

                // Add the registers horizontally to get one output
                LPGEMV_YMM2XMM( ymm16, ymm17, ymm18, ymm19, ymm0, ymm1, ymm2, ymm3, xmm0 );
                LPGEMV_YMM2XMM( ymm20, ymm21, ymm22, ymm23, ymm4, ymm5, ymm6, ymm7, xmm1 );

                // compose outputs into one ymm to perform post-ops
                ymm30 = _mm256_insertf128_ps( ymm30, xmm0, 0 );
                ymm30 = _mm256_insertf128_ps( ymm30, xmm1, 1 );
                is_mr_8 = 1;
            }
            if ( mr0_use >= 4 )
            {
                for( dim_t k = 0; k < k_iter; k++ )
                {
                    ymm15 = _mm256_loadu_ps( b_use );
                    b_use += 8; // move b pointer to next 8 elements

                    // Load 4x8 from row 0-3 of A
                    LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a );

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm24, ymm25, ymm26, ymm27, ymm15, ymm0, ymm1, ymm2, ymm3 );

                    a_use += 8;
                } // k-loop

                if( k_rem )
                {
                    ymm15 = _mm256_maskload_ps( b_use, k_rem_mask );

                    // Load 4x8 from row 0-3 of A
                    LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, k_rem_mask, a_use, rs_a );

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm24, ymm25, ymm26, ymm27, ymm15, ymm0, ymm1, ymm2, ymm3 );
                }

                //update pointers
                mr0_use -= 4;
                a_use = a_use_fringe + 4 * rs_a;
                a_use_fringe = a_use;
                b_use = b;

                // Add the registers horizontally to get one output
                LPGEMV_YMM2XMM( ymm24, ymm25, ymm26, ymm27, ymm0, ymm1, ymm2, ymm3, xmm0 );

                // compose outputs into one ymm to perform post-ops
                ymm31 = _mm256_insertf128_ps( ymm31, xmm0, 0 );
                regidx = 1;
            }
            if( mr0_use )
            {
                if( mr0_use >= 2 )
                {
                    for( dim_t k = 0; k < k_iter; k++ )
                    {
                        ymm15 = _mm256_loadu_ps( b_use );
                        b_use += 8; // move b pointer to next 8 elements

                        // Load 2x8 from row 0-1 of A
                        ymm0 = _mm256_loadu_ps( a_use );
                        ymm1 = _mm256_loadu_ps( a_use + rs_a );
                        a_use += 8;

                        // Perform the dot product
                        ymm28 = _mm256_fmadd_ps( ymm0, ymm15, ymm28 );
                        ymm29 = _mm256_fmadd_ps( ymm1, ymm15, ymm29 );
                    } // k-loop

                    if( k_rem )
                    {
                        ymm15 = _mm256_maskload_ps( b_use, k_rem_mask );

                        // Load 4x8 from row 0-3 of A
                        ymm0 = _mm256_maskload_ps( a_use, k_rem_mask );
                        ymm1 = _mm256_maskload_ps( a_use + rs_a, k_rem_mask );

                        // Perform the dot product
                        ymm28 = _mm256_fmadd_ps( ymm0, ymm15, ymm28 );
                        ymm29 = _mm256_fmadd_ps( ymm1, ymm15, ymm29 );
                    }

                    //update pointers
                    mr0_use -= 2;
                    a_use = a_use_fringe + 2 * rs_a;
                    a_use_fringe = a_use;
                    b_use = b;
                }
                if( mr0_use == 1 )
                {
                    for( dim_t k = 0; k < k_iter; k++ )
                    {
                        ymm15 = _mm256_loadu_ps( b_use );
                        b_use += 8; // move b pointer to next 8 elements

                        // Load 1x8 from row 0 of A
                        ymm0 = _mm256_loadu_ps( a_use );
                        a_use += 8;

                        // Perform the dot product
                        ymm14 = _mm256_fmadd_ps( ymm0, ymm15, ymm14 );
                    } // k-loop

                    if( k_rem )
                    {
                        ymm15 = _mm256_maskload_ps( b_use, k_rem_mask );

                        // Load 4x8 from row 0-3 of A
                        ymm0 = _mm256_maskload_ps( a_use, k_rem_mask );

                        // Perform the dot product
                        ymm14 = _mm256_fmadd_ps( ymm0, ymm15, ymm14 );
                    }
                    // When only fringe 1, update the registers to store in order
                    if (!(mr0 & 0x2))  ymm28 = ymm14;
                }

                LPGEMV_YMM2XMM( ymm28, ymm29, ymm14, ymm13, ymm0, ymm1, ymm2, ymm3, xmm1 );
                if( regidx == 0 ) ymm31 = _mm256_insertf128_ps( ymm31, xmm1, 0 );
                else ymm31 = _mm256_insertf128_ps( ymm31, xmm1, 1 );
            }

            if( !is_mr_8 ) ymm30 = ymm31;
        }

        // scale accumulated output with alpha
        ymm0 = _mm256_set1_ps( alpha );

        ymm30 = _mm256_mul_ps( ymm30, ymm0 );
        ymm31 = _mm256_mul_ps( ymm31, ymm0 );

        if( beta != 0.0f )
        {
            const float *_cbuf = c_use;

            //C = beta*C + alpha*A*B
            ymm3 = _mm256_set1_ps(beta);
            if( rs_c == 1 )
            {
                ymm0 = _mm256_maskload_ps( _cbuf, store_mask1 );
                ymm1 = _mm256_maskload_ps( _cbuf + 8, store_mask2 );
            }
            else
            {
                float ctemp[16] = {0};
                for( dim_t i = 0; i < mr0; i++ )
                {
                    ctemp[i] = _cbuf[i * rs_c];
                }
                ymm0 = _mm256_maskload_ps( ctemp, store_mask1 );
                ymm1 = _mm256_maskload_ps( ctemp + 8, store_mask2 );
            }

            // scale C with beta
            ymm30 = _mm256_fmadd_ps( ymm3, ymm0, ymm30 );
            ymm31 = _mm256_fmadd_ps( ymm3, ymm1, ymm31 );
        }

        // post-ops
        post_ops_attr.is_last_k = TRUE;
        lpgemm_post_op *post_ops_list_temp = post_op;
        POST_OP_LABEL_LASTK_SAFE_JUMP


POST_OPS_BIAS_1x32F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
               ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if( post_ops_list_temp->stor_type == BF16 )
                {
                    BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
                    BF16_F32_BIAS_BCAST_AVX2(ymm1,0);
                }
                else
                {
                    ymm0 = _mm256_set1_ps(*( ( float * )post_ops_list_temp->op_args1 ) );
                    ymm1 = _mm256_set1_ps(*( ( float * )post_ops_list_temp->op_args1 ) );
                }
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
                    __m128i bias_mask1 = _mm256_cvtepi32_epi16(store_mask1);
                    __m128i bias_mask2 = _mm256_cvtepi32_epi16(store_mask2);
                    ymm0 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                        _mm_maskload_epi32(
                          ( int const* )( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
                          post_ops_attr.post_op_c_i )
                        , bias_mask1 ) ), _mm256_set1_epi32( 16 ) )
                        );
                    ymm1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                        _mm_maskload_epi32(
                            ( int const* )( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
                            post_ops_attr.post_op_c_i + 8)
                        , bias_mask2 ) ), _mm256_set1_epi32( 16 ) )
                        );
                }
                else
                {
                    ymm0 =  _mm256_maskload_ps( ( float* )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_i , store_mask1 );
                    ymm1 =  _mm256_maskload_ps( ( float* )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_i + 8, store_mask2 );
                }
            }
            ymm30 = _mm256_add_ps(ymm0, ymm30);
            ymm31 = _mm256_add_ps(ymm1, ymm31);

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_RELU_1x32F:
        {
            ymm0 = _mm256_setzero_ps();
            ymm30 = _mm256_max_ps( ymm30, ymm0 );
            ymm31 = _mm256_max_ps( ymm31, ymm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_RELU_SCALE_1x32F:
        {
            ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            ymm1 = _mm256_setzero_ps();

            RELU_SCALE_OP_F32S_AVX2( ymm30, ymm0, ymm1, ymm2 );
            RELU_SCALE_OP_F32S_AVX2( ymm31, ymm0, ymm1, ymm2 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_GELU_TANH_1x32F:
        {
            __m256 dn, x_tanh;
            __m256i q;

            GELU_TANH_F32_AVX2_DEF(ymm30, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32_AVX2_DEF(ymm31, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_GELU_ERF_1x32F:
        {
          GELU_ERF_F32S_AVX2(ymm30, ymm0, ymm1, ymm2)
          GELU_ERF_F32S_AVX2(ymm31, ymm0, ymm1, ymm2)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_CLIP_1x32F:
        {
            ymm0 = _mm256_set1_ps(*(float *)post_ops_list_temp->op_args2);
            ymm1 = _mm256_set1_ps(*(float *)post_ops_list_temp->op_args3);

            CLIP_F32S_AVX2(ymm30, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm31, ymm0, ymm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_DOWNSCALE_1x32F:
        {
            __m256 zero_point0 = _mm256_setzero_ps();
            __m256 zero_point1 = _mm256_setzero_ps();
            __m256 selector1 = _mm256_setzero_ps();
            __m256 selector2 = _mm256_setzero_ps();

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
                selector1 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            else
            {

            }
            if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
                }
                else
                {
                    zero_point0 = _mm256_set1_ps( *(float *)post_ops_list_temp->op_args1 );
                    zero_point1 = _mm256_set1_ps( *(float *)post_ops_list_temp->op_args1 );
                }
            }
            else
            {
                // If original output was columns major, then by the time
                // kernel sees it, the matrix would be accessed as if it were
                // transposed. Due to this the scale as well as zp array will
                // be accessed by the ic index, and each scale/zp element
                // corresponds to an entire row of the transposed output array,
                // instead of an entire column.
            }
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                // Scale/zp len cannot be > 1, since orignal n = 1.
                F32_SCL_MULRND_AVX2(ymm30, selector1, zero_point0);
                F32_SCL_MULRND_AVX2(ymm31, selector2, zero_point1);
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
                    selector1 = _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i, store_mask1 );
                    selector2 = _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i + 8, store_mask2 );
                }
                if( *( dim_t*)post_ops_list_temp->op_args3 > 1 )
                {
                    __m128i zp_mask1 = _mm256_cvtepi32_epi16(store_mask1);
                    __m128i zp_mask2 = _mm256_cvtepi32_epi16(store_mask2);
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_BIAS_LOAD_AVX2_MASK_GEMV(zero_point0,0,zp_mask1)
                        BF16_F32_BIAS_LOAD_AVX2_MASK_GEMV(zero_point1,1,zp_mask2)
                    }
                    else
                    {
                        zero_point0 = _mm256_maskload_ps( ( float * )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_i, store_mask1 );
                        zero_point1 = _mm256_maskload_ps( ( float * )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_i + 8, store_mask2 );
                    }
                }
                F32_SCL_MULRND_AVX2(ymm30, selector1, zero_point0);
                F32_SCL_MULRND_AVX2(ymm31, selector2, zero_point1);
            }

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_MATRIX_ADD_1x32F:
        {
          __m256 selector1 = _mm256_setzero_ps();
          __m256 selector2 = _mm256_setzero_ps();
          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

          __m256 scl_fctr1 = _mm256_setzero_ps();
          __m256 scl_fctr2 = _mm256_setzero_ps();

          bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

          // Even though different registers are used for scalar in column and
          // row major case, all those registers will contain the same value.
          // For column major, if m==1, then it means n=1 and scale_factor_len=1.
          if ( post_ops_list_temp->scale_factor_len == 1 )
          {
            scl_fctr1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            scl_fctr2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          }
          else
          {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'c' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'C' ) )
            {
              scl_fctr1 =
                _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                    post_ops_attr.post_op_c_i + ( 0 * 8 ), store_mask1 );
              scl_fctr2 =
                _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                    post_ops_attr.post_op_c_i + ( 1 * 8 ), store_mask2 );
            }
          }
          if ( is_bf16 == TRUE )
          {
            bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

            if( ldm == 1 )
            {
                selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_load_si128(
                                ( __m128i const* )( matptr + post_ops_attr.post_op_c_i ) ) ),
                                _mm256_set1_epi32( 16 ) )
                            );

                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
                selector2 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_load_si128(
                                ( __m128i const* )( matptr + post_ops_attr.post_op_c_i + 8 ) ) ),
                                _mm256_set1_epi32( 16 ) )
                            );
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_add_ps( selector1, ymm30 );
                ymm31 = _mm256_add_ps( selector2, ymm31 );
            }
            else
            {
                bfloat16 ctemp[16];
                __m128i matstore_mask1 = _mm256_cvtepi32_epi16(store_mask1);
                __m128i matstore_mask2 = _mm256_cvtepi32_epi16(store_mask2);

                for( dim_t i = 0; i < mr0; i++ )
                {
                    ctemp[i] = *( matptr +
                                ( ( post_ops_attr.post_op_c_i + i )
                                    * ldm ) );
                }
                selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_maskload_epi32( ( int const* )( ctemp ),
                                matstore_mask1 ) ), _mm256_set1_epi32( 16 ) )
                            );
                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
                selector2 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_maskload_epi32( ( int const* )( ctemp + 8),
                                matstore_mask2 ) ), _mm256_set1_epi32( 16 ) )
                            );
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_add_ps( selector1, ymm30 );
                ymm31 = _mm256_add_ps( selector2, ymm31 );
            }
          }
          else
          {
            float* matptr = ( float* )post_ops_list_temp->op_args1;

            if( ldm == 1 )
            {
                selector1 = _mm256_maskload_ps(( matptr +
                                            post_ops_attr.post_op_c_i ), store_mask1 );
                selector2 = _mm256_maskload_ps(( matptr +
                                            post_ops_attr.post_op_c_i + 8 ), store_mask2 );

                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_add_ps( selector1, ymm30 );
                ymm31 = _mm256_add_ps( selector2, ymm31 );
            }
            else
            {
                float ctemp[16] = {0};
                for( dim_t i = 0; i < mr0; i++ )
                {
                    ctemp[i] = *( matptr +
                                ( ( post_ops_attr.post_op_c_i + i )
                                    * ldm ) );
                }
                selector1 = _mm256_maskload_ps( ctemp, store_mask1 );
                selector2 = _mm256_maskload_ps( ctemp + 8, store_mask2 );

                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_add_ps( selector1, ymm30 );
                ymm31 = _mm256_add_ps( selector2, ymm31 );
            }
        }
          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_MATRIX_MUL_1x32F:
       {
         __m256 selector1 = _mm256_setzero_ps();
         __m256 selector2 = _mm256_setzero_ps();

         dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

         __m256 scl_fctr1 = _mm256_setzero_ps();
         __m256 scl_fctr2 = _mm256_setzero_ps();

         bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

         // Even though different registers are used for scalar in column and
         // row major case, all those registers will contain the same value.
         // For column major, if m==1, then it means n=1 and scale_factor_len=1.
         if ( post_ops_list_temp->scale_factor_len == 1 )
         {
           scl_fctr1 =
             _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
           scl_fctr2 =
             _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
         }
         else
         {
           if ( ( *( char* )post_ops_list_temp->op_args2 == 'c' ) ||
               ( *( char* )post_ops_list_temp->op_args2 == 'C' ) )
           {
             scl_fctr1 =
               _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                   post_ops_attr.post_op_c_i + ( 0 * 16 ), store_mask1 );
             scl_fctr2 =
                _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                     post_ops_attr.post_op_c_i + ( 1 * 16 ), store_mask2 );
           }
         }
         if ( is_bf16 == TRUE )
          {
            bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

            if( ldm == 1 )
            {
                selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                            _mm_load_si128(
                            ( __m128i const* )( matptr + post_ops_attr.post_op_c_i ) ) ),
                            _mm256_set1_epi32( 16 ) )
                        );

                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
                selector2 =( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_load_si128(
                                ( __m128i const* )( matptr + post_ops_attr.post_op_c_i + 8 ) ) ),
                                _mm256_set1_epi32( 16 ) )
                            );
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_mul_ps( selector1, ymm30 );
                ymm31 = _mm256_mul_ps( selector2, ymm31 );
            }
            else
            {
                bfloat16 ctemp[16];
                __m128i matstore_mask1 = _mm256_cvtepi32_epi16(store_mask1);
                __m128i matstore_mask2 = _mm256_cvtepi32_epi16(store_mask2);

                for( dim_t i = 0; i < mr0; i++ )
                {
                    ctemp[i] = *( matptr +
                                ( ( post_ops_attr.post_op_c_i + i )
                                    * ldm ) );
                }
                selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_maskload_epi32( ( int const* )( ctemp ),
                                matstore_mask1 ) ), _mm256_set1_epi32( 16 ) )
                            );
                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
                selector2 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_maskload_epi32( ( int const* )( ctemp + 8),
                                matstore_mask2 ) ), _mm256_set1_epi32( 16 ) )
                            );
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_mul_ps( selector1, ymm30 );
                ymm31 = _mm256_mul_ps( selector2, ymm31 );
            }
          }
          else
          {
            float* matptr = ( float* )post_ops_list_temp->op_args1;

            if( ldm == 1 )
            {
                selector1 = _mm256_maskload_ps(( matptr +
                                                post_ops_attr.post_op_c_i ), store_mask1 );
                selector2 = _mm256_maskload_ps(( matptr +
                                                post_ops_attr.post_op_c_i + 8 ), store_mask2 );

                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_mul_ps( selector1, ymm30 );
                ymm31 = _mm256_mul_ps( selector2, ymm31 );
            }
            else
            {
                float ctemp[16];
                for( dim_t i = 0; i < mr0; i++ )
                {
                    ctemp[i] = *( matptr +
                                ( ( post_ops_attr.post_op_c_i + i )
                                    * ldm ) );
                }
                selector1 = _mm256_maskload_ps( ctemp, store_mask1 );
                selector2 = _mm256_maskload_ps( ctemp + 8, store_mask2 );

                selector1 = _mm256_mul_ps( selector1, scl_fctr1 ); \
                selector2 = _mm256_mul_ps( selector2, scl_fctr2 );

                ymm30 = _mm256_mul_ps( selector1, ymm30 );
                ymm31 = _mm256_mul_ps( selector2, ymm31 );
            }
        }
         POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_SWISH_1x32F:
       {
         ymm7 =
             _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
         __m256i ex_out;

         SWISH_F32_AVX2_DEF(ymm30, ymm7, ymm0, ymm1, ymm2, ymm3, ymm4, ex_out);
         SWISH_F32_AVX2_DEF(ymm31, ymm7, ymm0, ymm1, ymm2, ymm3, ymm4, ex_out);

         POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_TANH_1x32F:
       {
         __m256i ymm6;
         // c[0, 0-15]
         TANH_F32S_AVX2(ymm30, ymm0, ymm1, ymm2, ymm3, ymm4, ymm6)
         TANH_F32S_AVX2(ymm31, ymm0, ymm1, ymm2, ymm3, ymm4, ymm6)

         POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_SIGMOID_1x32F:
       {
         __m256i ex_out;

         // c[0, 0-15]
         SIGMOID_F32_AVX2_DEF(ymm30, ymm0, ymm1, ymm2, ymm3, ymm4, ex_out);
         SIGMOID_F32_AVX2_DEF(ymm31, ymm0, ymm1, ymm2, ymm3, ymm4, ex_out);

         POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }

POST_OPS_1x32F_DISABLE:
        {        // store the result
            if( rs_c == 1 )
            {
                // store the result
                _mm256_maskstore_ps( c_use, store_mask1, ymm30 );
                _mm256_maskstore_ps( c_use + 8, store_mask2, ymm31 );
            }
            else
            {
                float ctemp[16];
                _mm256_storeu_ps( ctemp, ymm30 );
                _mm256_storeu_ps( ctemp + 8, ymm31 );
                for( dim_t i = 0; i < mr0; i++ )
                {
                    c_use[i * rs_c] = ctemp[i];
                }
            }
        }

        post_ops_attr.post_op_c_i += MR;

    } // mr loop
}
#endif // BLIS_ADDON_LPGEMM

