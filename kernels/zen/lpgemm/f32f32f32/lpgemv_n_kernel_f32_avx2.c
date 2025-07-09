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

// When n=1 is load 16x1 from B and load MRx16 from A and perform dot product
//  to produce C output of MRX1. The vectorization is done in k loop and
//  the horizontal reduction done to produce one output from each
//  accumulator register

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

LPGEMV_N_EQ1_KERN( float, float, float, f32f32f32of32_avx2 )
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

    // Strides are updated based on matrix packing/reordering.
    const float *a_use = NULL;
    const float *b_use = NULL;
    float *c_use = NULL;

    lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1;

    static const int32_t mask[9][9] = {
                    {0, 0, 0, 0, 0, 0, 0, 0}, //load no values, not used currently
                    {-1, 0, 0, 0, 0, 0, 0, 0}, // load 1 value from memory
                    {-1, -1, 0, 0, 0, 0, 0, 0}, // load 2 values from memory
                    {-1, -1, -1, 0, 0, 0, 0, 0},
                    {-1, -1, -1, -1, 0, 0, 0, 0},
                    {-1, -1, -1, -1, -1, 0, 0, 0},
                    {-1, -1, -1, -1, -1, -1, 0, 0},
                    {-1, -1, -1, -1, -1, -1, -1, 0},
                    {-1, -1, -1, -1, -1, -1, -1, -1}
                  };

    // MR comes from framework, we need to set it based on the underlying hardware configuration.
    for (dim_t mr = 0; mr < m0; mr += MR)
    {
        dim_t mr0 = bli_min( m0 - mr, MR );

        dim_t k_iter = k / 8;
        dim_t k_rem = k % 8;

        __m256i store_mask = _mm256_loadu_si256((__m256i*)mask[mr0]);
        __m256i k_rem_mask = _mm256_loadu_si256((__m256i*)mask[k_rem]);

        /* zero the accumulator registers */
        ZERO_ACC_YMM_4_REG( ymm8, ymm9, ymm10, ymm11 );
        ZERO_ACC_YMM_4_REG( ymm12, ymm13, ymm14, ymm15 );

        //update pointers
        a_use = a + mr * rs_a;
        b_use = b;
        c_use = c + mr * rs_c;

        if( mr0 == MR )
        {
            for( dim_t k = 0; k < k_iter; k++ )
            {
                ymm7 = _mm256_loadu_ps( b_use );
                b_use += 8; // move b pointer to next 8 elements

                // Load 4x8 from row 0-3 of A
                LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm8, ymm9, ymm10, ymm11, ymm7, ymm0, ymm1, ymm2, ymm3 );

                // Load 4x8 from row 4-7 of A
                LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a );
                a_use -= 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm12, ymm13, ymm14, ymm15, ymm7, ymm0, ymm1, ymm2, ymm3 );

                a_use += 8;
            } // k-loop

            if( k_rem )
            {
                ymm7 = _mm256_maskload_ps( b_use, k_rem_mask );

                // Load 4x8 from row 0-3 of A
                LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, k_rem_mask, a_use, rs_a );
                a_use += 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm8, ymm9, ymm10, ymm11, ymm7, ymm0, ymm1, ymm2, ymm3 );

                // Load 4x8 from row 4-7 of A
                LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, k_rem_mask, a_use, rs_a );
                a_use -= 4 * rs_a; // move a pointer to next 4x8 elements

                // Perform the dot product
                LPGEMV_N_KERNEL_4_FMA( ymm12, ymm13, ymm14, ymm15, ymm7, ymm0, ymm1, ymm2, ymm3 );
            }

            // Add the registers horizontally to get one output
            LPGEMV_YMM2XMM( ymm8, ymm9, ymm10, ymm11, ymm0, ymm1, ymm2, ymm3, xmm0 );
            LPGEMV_YMM2XMM( ymm12, ymm13, ymm14, ymm15, ymm4, ymm1, ymm2, ymm3, xmm1 );

            // compose outputs into one ymm to perform post-ops.
            ymm8 = _mm256_insertf128_ps( ymm8, xmm0, 0 );
            ymm8 = _mm256_insertf128_ps( ymm8, xmm1, 1 );
        }
        else
        {
            //Handle fringe cases when mr0 < MR
            const float *a_use_fringe = a_use;
            dim_t mr0_use = mr0;
            dim_t regidx = 0;

            // Dot product for mfringe 4
            if (mr0_use >= 4)
            {
                for( dim_t k = 0; k < k_iter; k++ )
                {
                    ymm7 = _mm256_loadu_ps( b_use );
                    b_use += 8; // move b pointer to next 8 elements

                    // Load 4x8 from row 0-3 of A
                    LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a );
                    a_use += 8; // move a pointer to next 4x8 elements

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm8, ymm9, ymm10, ymm11, ymm7, ymm0, ymm1, ymm2, ymm3 );
                } // k-loop

                if( k_rem )
                {
                    ymm7 = _mm256_maskload_ps( b_use, k_rem_mask );

                    // Load 4x8 from row 0-3 of A
                    LPGEMV_N_KERNEL_4_MASKLOADS( ymm0, ymm1, ymm2, ymm3, k_rem_mask, a_use, rs_a );

                    // Perform the dot product
                    LPGEMV_N_KERNEL_4_FMA( ymm8, ymm9, ymm10, ymm11, ymm7, ymm0, ymm1, ymm2, ymm3 );
                }

                //update pointers
                mr0_use -= 4;
                a_use = a_use_fringe + 4 * rs_a;
                a_use_fringe = a_use;
                b_use = b;

                //Horizontal add 4 ymm registers and get output into 2 xmm registers
                LPGEMV_YMM2XMM(ymm8, ymm9, ymm10, ymm11, ymm0, ymm1, ymm2, ymm3, xmm0)

                // compose outputs into one ymm to perform post-ops.
                ymm8 = _mm256_insertf128_ps( ymm8, xmm0, 0 );
                regidx = 1;
            }
            // Dot product for  <= 3
            if (mr0_use)
            {
                if( mr0_use >= 2 )
                {
                    for (dim_t k = 0; k < k_iter; k++)
                    {
                        ymm7 = _mm256_loadu_ps( b_use );
                        b_use += 8; // move b pointer to next 8 elements

                        // Load 2x16 from row 0-1 of A
                        ymm0 = _mm256_loadu_ps( a_use );
                        ymm1 = _mm256_loadu_ps( a_use + rs_a );
                        a_use += 8; // move a pointer to next 4x8 elements

                        ymm12 = _mm256_fmadd_ps( ymm0, ymm7, ymm12 );
                        ymm13 = _mm256_fmadd_ps( ymm1, ymm7, ymm13 );

                    } // k-loop
                    if( k_rem )
                    {
                        ymm7 = _mm256_maskload_ps( b_use, k_rem_mask );

                        // Load 2x16 from row 0-1 of A
                        ymm0 = _mm256_maskload_ps( a_use, k_rem_mask );
                        ymm1 = _mm256_maskload_ps( a_use + rs_a, k_rem_mask );

                        ymm12 = _mm256_fmadd_ps( ymm0, ymm7, ymm12 );
                        ymm13 = _mm256_fmadd_ps( ymm1, ymm7, ymm13 );
                    }
                    //update pointers
                    mr0_use -= 2;
                    a_use = a_use_fringe + 2 * rs_a;
                    a_use_fringe = a_use;
                    b_use = b;
                }
                if( mr0_use == 1 )
                {
                    for (dim_t k = 0; k < k_iter; k++)
                    {
                        ymm7 = _mm256_loadu_ps( b_use );
                        b_use += 8; // move b pointer to next 8 elements

                        // Load 1x16 from row 0 of A
                        ymm0 = _mm256_loadu_ps( a_use );
                        a_use += 8; // move a pointer to next 4x8 elements

                        ymm14 = _mm256_fmadd_ps( ymm0, ymm7, ymm14 );

                    } // k-loop
                    if( k_rem )
                    {
                        ymm7 = _mm256_maskload_ps( b_use, k_rem_mask );

                        // Load 1x16 from row 0 of A
                        ymm0 = _mm256_maskload_ps( a_use, k_rem_mask );

                        ymm14 = _mm256_fmadd_ps( ymm0, ymm7, ymm14 );
                    }
                    // When only fringe 1, update the registers to store in order
                    if (!(mr0 & 0x2))  ymm12 = ymm14;
                }

                LPGEMV_YMM2XMM( ymm12, ymm13, ymm14, ymm15, ymm0, ymm1, ymm2, ymm3, xmm1 );
                if (regidx == 0) ymm8 = _mm256_insertf128_ps(ymm8, xmm1, 0);
                else ymm8 = _mm256_insertf128_ps(ymm8, xmm1, 1);
            }
        }

        // scale accumulated output with alpha
        ymm0 = _mm256_set1_ps( alpha );
        ymm8 = _mm256_mul_ps( ymm8, ymm0 );

        if( beta != 0.0f )
        {
            const float *_cbuf = c_use;

            //C = beta*C + alpha*A*B
            ymm3 = _mm256_set1_ps(beta);
            if( rs_c == 1 )
            {
              if ( post_ops_attr.buf_downscale != NULL )
                {
                    ymm0 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                        _mm_loadu_si128(
                          ( __m128i const* )( ( ( bfloat16* )post_ops_attr.buf_downscale ) +
                          ( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + 0 ) )
                          + post_ops_attr.post_op_c_j + (0 * 8) ) ) ), _mm256_set1_epi32( 16 ) )
                        );
                }
                else
                {
                    ymm0 = _mm256_maskload_ps( _cbuf, store_mask );
                }
            }
            else
            {
              if ( post_ops_attr.buf_downscale != NULL  )
                {
                    bfloat16 ctemp[8] = {0};
                    for( dim_t i = 0; i < mr0; i++ )
                    {
                        ctemp[i] = *( ( bfloat16* )post_ops_attr.buf_downscale +
                                    ( post_ops_attr.rs_c_downscale *
                                    ( post_ops_attr.post_op_c_i + i ) ) );
                    }
                    ymm0 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                                _mm_loadu_si128(
                                ( __m128i const* )( (bfloat16* )ctemp) ) ),
                                _mm256_set1_epi32( 16 ) )
                            );
                }
                else
                {
                    // load c into ymm0
                    float ctemp[8] = { 0 };
                    for( dim_t i = 0; i < mr0; i++ )
                    {
                        ctemp[i] = _cbuf[i * rs_c];
                    }
                    ymm0 = _mm256_loadu_ps( ctemp );
                }
            }

            // scale c with beta
            ymm8 = _mm256_fmadd_ps( ymm0, ymm3, ymm8 );
        }

        // post-ops
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
                ymm0 = (__m256)( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                      _mm_set1_epi16(
                         *( ( bfloat16* )post_ops_list_temp->op_args1 )
                      ) ), _mm256_set1_epi32( 16 ) )
                      );
              }
              else
              {
                ymm0 = _mm256_set1_ps(*( ( float * )post_ops_list_temp->op_args1 ) );
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
             __m128i bias_mask = _mm_loadu_si128((__m128i*)mask[mr0]);
              ymm0 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                    _mm_maskload_epi32(
                      ( int const* )( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
                      post_ops_attr.post_op_c_i )
                    , bias_mask ) ), _mm256_set1_epi32( 16 ) )
                    );
            }
            else
            {
              ymm0 =  _mm256_maskload_ps( ( float* )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i , store_mask );
            }
          }
          ymm8 = _mm256_add_ps(ymm0, ymm8);
          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_RELU_1x16F:
        {
            ymm0 = _mm256_setzero_ps();
            ymm8 = _mm256_max_ps( ymm8, ymm0 );
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_RELU_SCALE_1x16F:
        {
            ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            ymm1 = _mm256_setzero_ps();

            RELU_SCALE_OP_F32S_AVX2( ymm8, ymm0, ymm1, ymm2 );
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_GELU_TANH_1x16F:
        {
            __m256 dn, x_tanh;
            __m256i q;

            // c[0,0-3]
            GELU_TANH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_GELU_ERF_1x16F:
        {
          // c[0, 0-15]
          GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)
          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_CLIP_1x16F:
        {
          ymm0 = _mm256_set1_ps(*(float *)post_ops_list_temp->op_args2);
          ymm1 = _mm256_set1_ps(*(float *)post_ops_list_temp->op_args3);

          // c[0, 0-15]
          CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_DOWNSCALE_1x16F:
        {
            __m256 zero_point0 = _mm256_setzero_ps();
            __m256 selector1 = _mm256_setzero_ps();

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

            }
            if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
              if( is_bf16 == TRUE )
              {
                BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
              }
              else
              {
                zero_point0 = _mm256_set1_ps( *(float *)post_ops_list_temp->op_args1 );
              }
            }
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
              // Scale/zp len cannot be > 1, since orignal n = 1.
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
                selector1 = _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                                    post_ops_attr.post_op_c_i, store_mask );
            }
            if( *( dim_t*)post_ops_list_temp->op_args3 > 1 )
            {
              if( is_bf16 == TRUE )
              {
                __m128i zp_mask = _mm_loadu_si128((__m128i*)mask[mr0]);
                zero_point0 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                  _mm_maskload_epi32(
                    ( int const* )( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
                    post_ops_attr.post_op_c_i )
                  , zp_mask ) ), _mm256_set1_epi32( 16 ) )
                  );
              }
              else
              {
                zero_point0 = _mm256_maskload_ps( ( float * )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i, store_mask );
              }
            }
              F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_MATRIX_ADD_1x16F:
        {
          __m256 selector1 = _mm256_setzero_ps();

          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

          __m256 scl_fctr1 = _mm256_setzero_ps();

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
          }
          else
          {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'c' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'C' ) )
            {
              scl_fctr1 =
                _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                    post_ops_attr.post_op_c_i + ( 0 * 16 ), store_mask );
            }
          }
          if ( is_bf16 == TRUE )
          {
            bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
            __m128i _mask = _mm_loadu_si128((__m128i*)mask[mr0]);

            if( ldm == 1 )
            {
              selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                  _mm_maskload_epi32(
                    ( int const* )( matptr + post_ops_attr.post_op_c_i )
                  , _mask ) ), _mm256_set1_epi32( 16 ) )
                );

              selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
              ymm8 = _mm256_add_ps( selector1, ymm8 );
            }
            else
            {
              bfloat16 ctemp[16];
              for( dim_t i = 0; i < mr0; i++ )
              {
                ctemp[i] = *( matptr +
                            ( ( post_ops_attr.post_op_c_i + i )
                                * ldm ) );
              }
              selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                  _mm_maskload_epi32(
                  ( int const* )( ctemp ), _mask ) ), _mm256_set1_epi32( 16 ) )
              );
              selector1 = _mm256_mul_ps( selector1, scl_fctr1 ); \
              ymm8 = _mm256_add_ps( selector1, ymm8 );
            }
          }
          else
          {
            float* matptr = ( float* )post_ops_list_temp->op_args1;

            if( ldm == 1 )
            {
              selector1 = _mm256_maskload_ps(( matptr +
                                          post_ops_attr.post_op_c_i ), store_mask );
                selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
              ymm8 = _mm256_add_ps( selector1, ymm8 );

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
              selector1 = _mm256_maskload_ps( ctemp, store_mask );
              selector1 = _mm256_mul_ps( selector1, scl_fctr1 ); \
              ymm8 = _mm256_add_ps( selector1, ymm8 );
            }
          }
          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_MATRIX_MUL_1x16F:
       {
         __m256 selector1 = _mm256_setzero_ps();

         dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

         __m256 scl_fctr1 = _mm256_setzero_ps();

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
         }
         else
         {
           if ( ( *( char* )post_ops_list_temp->op_args2 == 'c' ) ||
               ( *( char* )post_ops_list_temp->op_args2 == 'C' ) )
           {
             scl_fctr1 =
               _mm256_maskload_ps( ( float* )post_ops_list_temp->scale_factor +
                   post_ops_attr.post_op_c_i + ( 0 * 16 ), store_mask );
           }
         }
         if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
          __m128i _mask = _mm_loadu_si128((__m128i*)mask[mr0]);

          if( ldm == 1 )
          {
            selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
              _mm_maskload_epi32(
                ( int const* )( matptr + post_ops_attr.post_op_c_i )
              , _mask ) ), _mm256_set1_epi32( 16 ) )
            );

            selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
            ymm8 = _mm256_mul_ps( selector1, ymm8 );
          }
          else
          {
            bfloat16 ctemp[16];
            for( dim_t i = 0; i < mr0; i++ )
            {
              ctemp[i] = *( matptr +
                          ( ( post_ops_attr.post_op_c_i + i )
                              * ldm ) );
            }
            selector1 = ( __m256 )( _mm256_sllv_epi32( _mm256_cvtepi16_epi32(
                      _mm_maskload_epi32(
                      ( int const* )( ctemp ), _mask ) ), _mm256_set1_epi32( 16 ) )
                    );
            selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
            ymm8 = _mm256_mul_ps( selector1, ymm8 );
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if( ldm == 1 )
          {
            selector1 = _mm256_maskload_ps(( matptr +
                                          post_ops_attr.post_op_c_i ), store_mask );
            selector1 = _mm256_mul_ps( selector1, scl_fctr1 );
            ymm8 = _mm256_mul_ps( selector1, ymm8 );
          }
          else
          {
            float ctemp[8];
            for( dim_t i = 0; i < mr0; i++ )
            {
              ctemp[i] = *( matptr +
                          ( ( post_ops_attr.post_op_c_i + i )
                              * ldm ) );
            }
            selector1 = _mm256_maskload_ps( ctemp, store_mask );
            selector1 = _mm256_mul_ps( selector1, scl_fctr1 ); \
            ymm8 = _mm256_mul_ps( selector1, ymm8 );
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_SWISH_1x16F:
       {
         ymm7 =
             _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
         __m256i ex_out;

         // c[0, 0-15]
         SWISH_F32_AVX2_DEF(ymm8, ymm7, ymm0, ymm1, ymm2, ymm3, ymm4, ex_out);

         POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_TANH_1x16F:
       {
         __m256i ymm6;
         // c[0, 0-15]
         TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, ymm4, ymm6)

         POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }
POST_OPS_SIGMOID_1x16F:
       {
         __m256i ex_out;

         // c[0, 0-15]
         SIGMOID_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, ymm4, ex_out);

         POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
       }

POST_OPS_1x16F_DISABLE:
        {
          if( ( post_ops_attr.buf_downscale != NULL ) &&
                    ( post_ops_attr.is_last_k == TRUE ) )
          {
              uint32_t tlsb, rounded, temp[8] = {0};
              int i;
              bfloat16* dest;

              if( rs_c == 1 )
              {
                  _mm256_maskstore_ps((float*)temp, store_mask, ymm8);

                  STORE_F32_BF16_N_ONE_YMM(temp, mr0)
              }
              else
              {
                  _mm256_storeu_ps((float*)temp, ymm8);

                  STORE_F32_BF16_N_ONE_YMM( temp, mr0 )
              }
          }
          else
          {
            if( rs_c == 1 )
            {
                _mm256_maskstore_ps ( c_use, store_mask, ymm8 );
            }
            else
            {
                // store c from ymm0
                float ctemp[8];
                _mm256_storeu_ps( ctemp, ymm8 );
                for( dim_t i = 0; i < mr0; i++ )
                {
                    c_use[i * rs_c] = ctemp[i];
                }
            }
          }
        }
        post_ops_attr.post_op_c_i += MR;
    } // mr loop
}
#endif // BLIS_ADDON_LPGEMM
