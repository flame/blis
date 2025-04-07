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

#include "lpgemm_kernel_macros_f32.h"
#include "../../../zen/lpgemm/f32f32f32/lpgemm_kernel_macros_f32_avx2.h"

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x64_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm4,  zmm6,  zmm8,  zmm9,
           zmm10, zmm11, zmm12, zmm13, zmm14, zmm15, zmm16, zmm17,
           zmm18, zmm19, zmm20, zmm21, zmm23, zmm24, zmm26, zmm27,
           zmm29, zmm30;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
           xmm8;

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // LOOP_5x64J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm3)
        ZERO_ACC_ZMM_4_REG( zmm4,  zmm6,  zmm8,  zmm9)
        ZERO_ACC_ZMM_4_REG(zmm10, zmm11, zmm12, zmm13)
        ZERO_ACC_ZMM_4_REG(zmm14, zmm15, zmm16, zmm17)
        ZERO_ACC_ZMM_4_REG(zmm18, zmm19, zmm20, zmm21)
        ZERO_ACC_ZMM_4_REG(zmm23, zmm24, zmm26, zmm27)
        ZERO_ACC_ZMM_2_REG(zmm29, zmm30)

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_4_REG(xmm4, xmm5, xmm6, xmm7)
        xmm8 = _mm_setzero_ps();

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
            zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
            zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
            zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
            zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
            zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm3  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);
            zmm4  = _mm512_maskz_loadu_ps(m_mask, a_temp + 4*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
            zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
            zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
            zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
            zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
        }


        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm21, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm24, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm27, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm30, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm8, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm11);
            xmm2 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm2);
            xmm2 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm7 = _mm_mul_ps(xmm3, xmm0);
            xmm8 = _mm_mul_ps(xmm2, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm21, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm24, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm27, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm30, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm8, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm11);
            xmm2 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm2);
            xmm2 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm3 = _mm_mul_ps(xmm3, xmm0);
            xmm2 = _mm_mul_ps(xmm2, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
            xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
            xmm1 = _mm_loadu_ps(c_temp + 4*rs_c0);
            xmm8 = _mm_fmadd_ps(xmm0, xmm1, xmm2);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x64F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_5x64F:
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

POST_OPS_RELU_SCALE_5x64F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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

POST_OPS_GELU_TANH_5x64F:
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

POST_OPS_GELU_ERF_5x64F:
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

POST_OPS_CLIP_5x64F:
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

POST_OPS_DOWNSCALE_5x64F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

                //c[3, 0-3]
                F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);

                //c[4, 0-3]
                F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
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
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 4 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,4)
                    }
                    else
                    {
                        zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                                                    post_ops_attr.post_op_c_i + 4 ) );
                    }
                }

                //c[4, 0-3]
                F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_MATRIX_ADD_5x64F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 3 ) );
                    scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_5x64F:
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

POST_OPS_SWISH_5x64F:
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

POST_OPS_TANH_5x64F:
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

POST_OPS_SIGMOID_5x64F:
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

POST_OPS_5x64F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm7);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm8);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp = c_temp + 6*rs_c0;
        a_temp = a_temp + 6*rs_a0;
    }   // END LOOP_5x64J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x64_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm6,  zmm8,  zmm9, zmm10,
           zmm11, zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18,
           zmm19, zmm20, zmm23, zmm26, zmm29;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // LOOP_4x64J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm3)
        ZERO_ACC_ZMM_4_REG( zmm6,  zmm8,  zmm9, zmm10)
        ZERO_ACC_ZMM_4_REG(zmm11, zmm12, zmm13, zmm14)
        ZERO_ACC_ZMM_4_REG(zmm15, zmm16, zmm17, zmm18)
        ZERO_ACC_ZMM_4_REG(zmm19, zmm20, zmm23, zmm26)
        zmm29 = _mm512_setzero_ps();

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_4_REG(xmm4, xmm5, xmm6, xmm7)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm3  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
        }


        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm7 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm7 = _mm_mul_ps(xmm7, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm3 = _mm_mul_ps(xmm3, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
            xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x64F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_4x64F:
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

POST_OPS_RELU_SCALE_4x64F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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

POST_OPS_GELU_TANH_4x64F:
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

POST_OPS_GELU_ERF_4x64F:
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

POST_OPS_CLIP_4x64F:
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

POST_OPS_DOWNSCALE_4x64F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

                //c[3, 0-3]
                F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
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
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_MATRIX_ADD_4x64F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_4x64F:
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

POST_OPS_SWISH_4x64F:
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

POST_OPS_TANH_4x64F:
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

POST_OPS_SIGMOID_4x64F:
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

POST_OPS_4x64F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm7);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_4x64J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x64_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm6,  zmm8,  zmm9, zmm10, zmm11,
           zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18, zmm19;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // LOOP_3x64J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm6)
        ZERO_ACC_ZMM_4_REG( zmm8,  zmm9, zmm10, zmm11)
        ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15)
        ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19)

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_3_REG(xmm4, xmm5, xmm6)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
        }


        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x64F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_3x64F:
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

POST_OPS_RELU_SCALE_3x64F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            // c[2,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_3x64F:
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

POST_OPS_GELU_ERF_3x64F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            // c[2,0-3]
            GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_3x64F:
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

POST_OPS_DOWNSCALE_3x64F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 1 ) );
                    selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_3x64F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_3x64F:
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

POST_OPS_SWISH_3x64F:
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

POST_OPS_TANH_3x64F:
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

POST_OPS_SIGMOID_3x64F:
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

POST_OPS_3x64F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_3x64J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x64_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm6,  zmm8, zmm9, zmm11, zmm12, zmm14,
           zmm15, zmm17, zmm18;

    __m256  ymm0,  ymm1,  ymm2, ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // LOOP_2x64J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm6,  zmm8)
        ZERO_ACC_ZMM_4_REG( zmm9, zmm11, zmm12, zmm14)
        ZERO_ACC_ZMM_3_REG(zmm15, zmm17, zmm18)

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x64F:
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

POST_OPS_RELU_2x64F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            // c[1,0-3]
            xmm5 = _mm_max_ps( xmm5, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_2x64F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_2x64F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            // c[1,0-3]
            GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_2x64F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_2x64F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            // c[1,0-3]
            CLIP_F32S_SSE(xmm5, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_2x64F:
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

POST_OPS_MATRIX_ADD_2x64F:
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

POST_OPS_MATRIX_MUL_2x64F:
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

POST_OPS_SWISH_2x64F:
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

POST_OPS_TANH_2x64F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            // c[1,0-3]
            TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_2x64F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            // c[1,0-3]
            SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_2x64F_DISABLE:
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

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_2x64J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x64_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512 zmm0, zmm6, zmm8, zmm11, zmm14, zmm17;

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm7, ymm10, ymm13;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // LOOP_1x64J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm6, zmm8, zmm11)
        ZERO_ACC_ZMM_2_REG(zmm14, zmm17)

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG(ymm0, ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG(ymm4, ymm7, ymm10, ymm13)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        xmm4 = _mm_setzero_ps();

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
        }


        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x64F:
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

POST_OPS_RELU_1x64F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_1x64F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_1x64F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_1x64F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_1x64F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_1x64F:
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

POST_OPS_MATRIX_ADD_1x64F:
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

POST_OPS_MATRIX_MUL_1x64F:
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

POST_OPS_SWISH_1x64F:
        {
            xmm0 =
                _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_TANH_1x64F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_1x64F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_1x64F_DISABLE:
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
            c_temp += rs_c;
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_1x64J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x48_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm4,  zmm6,  zmm8,  zmm9,
           zmm10, zmm11, zmm12, zmm13, zmm14, zmm15, zmm16, zmm17,
           zmm18, zmm19, zmm20, zmm21, zmm23, zmm24, zmm26, zmm27,
           zmm29, zmm30;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
           xmm8;

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // LOOP_5x48J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm3)
        ZERO_ACC_ZMM_4_REG( zmm4,  zmm6,  zmm8,  zmm9)
        ZERO_ACC_ZMM_4_REG(zmm10, zmm11, zmm12, zmm13)
        ZERO_ACC_ZMM_4_REG(zmm14, zmm15, zmm16, zmm17)
        ZERO_ACC_ZMM_4_REG(zmm18, zmm19, zmm20, zmm21)
        ZERO_ACC_ZMM_4_REG(zmm23, zmm24, zmm26, zmm27)
        ZERO_ACC_ZMM_2_REG(zmm29, zmm30)

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_4_REG(xmm4, xmm5, xmm6, xmm7)
        xmm8 = _mm_setzero_ps();

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
            zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
            zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
            zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
            zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
            zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm3  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);
            zmm4  = _mm512_maskz_loadu_ps(m_mask, a_temp + 4*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
            zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
            zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
            zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
            zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
        }


        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm21, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm24, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm27, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm30, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm8, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm11);
            xmm2 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm2);
            xmm2 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm7 = _mm_mul_ps(xmm3, xmm0);
            xmm8 = _mm_mul_ps(xmm2, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm21, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm24, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm27, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm30, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm8, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm11);
            xmm2 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm2);
            xmm2 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm3 = _mm_mul_ps(xmm3, xmm0);
            xmm2 = _mm_mul_ps(xmm2, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
            xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
            xmm1 = _mm_loadu_ps(c_temp + 4*rs_c0);
            xmm8 = _mm_fmadd_ps(xmm0, xmm1, xmm2);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x48F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_5x48F:
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

POST_OPS_RELU_SCALE_5x48F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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

POST_OPS_GELU_TANH_5x48F:
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

POST_OPS_GELU_ERF_5x48F:
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

POST_OPS_CLIP_5x48F:
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

POST_OPS_DOWNSCALE_5x48F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

                //c[3, 0-3]
                F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);

                //c[4, 0-3]
                F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
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
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 4 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,4)
                    }
                    else
                    {
                        zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                                                    post_ops_attr.post_op_c_i + 4 ) );
                    }
                }

                //c[4, 0-3]
                F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_MATRIX_ADD_5x48F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 3 ) );
                    scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_5x48F:
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

POST_OPS_SWISH_5x48F:
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

POST_OPS_TANH_5x48F:
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

POST_OPS_SIGMOID_5x48F:
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

POST_OPS_5x48F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm7);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm8);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp = c_temp + 6*rs_c0;
        a_temp = a_temp + 6*rs_a0;
    }   // END LOOP_5x48J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x48_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm6,  zmm8,  zmm9, zmm10,
           zmm11, zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18,
           zmm19, zmm20, zmm23, zmm26, zmm29;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // LOOP_4x48J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm3)
        ZERO_ACC_ZMM_4_REG( zmm6,  zmm8,  zmm9, zmm10)
        ZERO_ACC_ZMM_4_REG(zmm11, zmm12, zmm13, zmm14)
        ZERO_ACC_ZMM_4_REG(zmm15, zmm16, zmm17, zmm18)
        ZERO_ACC_ZMM_4_REG(zmm19, zmm20, zmm23, zmm26)
        zmm29 = _mm512_setzero_ps();

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_4_REG(xmm4, xmm5, xmm6, xmm7)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm3  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm7 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm7 = _mm_mul_ps(xmm7, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm3 = _mm_mul_ps(xmm3, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
            xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x48F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_4x48F:
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

POST_OPS_RELU_SCALE_4x48F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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

POST_OPS_GELU_TANH_4x48F:
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

POST_OPS_GELU_ERF_4x48F:
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

POST_OPS_CLIP_4x48F:
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

POST_OPS_DOWNSCALE_4x48F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

                //c[3, 0-3]
                F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
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
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_MATRIX_ADD_4x48F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_4x48F:
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

POST_OPS_SWISH_4x48F:
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

POST_OPS_TANH_4x48F:
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

POST_OPS_SIGMOID_4x48F:
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

POST_OPS_4x48F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm7);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_4x48J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x48_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm6,  zmm8,  zmm9, zmm10, zmm11,
           zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18, zmm19;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // LOOP_3x48J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm6)
        ZERO_ACC_ZMM_4_REG( zmm8,  zmm9, zmm10, zmm11)
        ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15)
        ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19)

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_3_REG(xmm4, xmm5, xmm6)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x48F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_3x48F:
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

POST_OPS_RELU_SCALE_3x48F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            // c[2,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_3x48F:
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

POST_OPS_GELU_ERF_3x48F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            // c[2,0-3]
            GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_3x48F:
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

POST_OPS_DOWNSCALE_3x48F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 1 ) );
                    selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_3x48F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_3x48F:
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

POST_OPS_SWISH_3x48F:
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

POST_OPS_TANH_3x48F:
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

POST_OPS_SIGMOID_3x48F:
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

POST_OPS_3x48F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_3x48J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x48_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm6, zmm8, zmm9, zmm11, zmm12, zmm14,
           zmm15, zmm17, zmm18;

    __m256  ymm0,  ymm1,  ymm2,  ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // LOOP_2x48J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm6,  zmm8)
        ZERO_ACC_ZMM_4_REG( zmm9, zmm11, zmm12, zmm14)
        ZERO_ACC_ZMM_3_REG(zmm15, zmm17, zmm18)

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG( ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x48F:
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

POST_OPS_RELU_2x48F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            // c[1,0-3]
            xmm5 = _mm_max_ps( xmm5, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_2x48F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_2x48F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            // c[1,0-3]
            GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_2x48F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_2x48F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            // c[1,0-3]
            CLIP_F32S_SSE(xmm5, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_2x48F:
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

POST_OPS_MATRIX_ADD_2x48F:
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

POST_OPS_MATRIX_MUL_2x48F:
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

POST_OPS_SWISH_2x48F:
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

POST_OPS_TANH_2x48F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            // c[1,0-3]
            TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_2x48F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            // c[1,0-3]
            SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_2x48F_DISABLE:
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

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_2x48J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x48_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512 zmm0, zmm6, zmm8, zmm11, zmm14, zmm17;

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm7, ymm10, ymm13;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // LOOP_1x48J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm6, zmm8, zmm11)
        ZERO_ACC_ZMM_2_REG(zmm14, zmm17)

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG(ymm0, ymm1, ymm2, ymm3)
        ZERO_ACC_YMM_4_REG(ymm4, ymm7, ymm10, ymm13)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        xmm4 = _mm_setzero_ps();

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x48F:
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

POST_OPS_RELU_1x48F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_1x48F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_1x48F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_1x48F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_1x48F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_1x48F:
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

POST_OPS_MATRIX_ADD_1x48F:
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

POST_OPS_MATRIX_MUL_1x48F:
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

POST_OPS_SWISH_1x48F:
        {
            xmm0 =
                _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_TANH_1x48F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_1x48F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_1x48F_DISABLE:
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
            c_temp += rs_c;
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_1x48J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x32_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512   zmm0,  zmm1,  zmm2,  zmm3,  zmm4,  zmm6,  zmm8, zmm9,
            zmm10, zmm11, zmm12, zmm13, zmm14, zmm15, zmm16, zmm17,
            zmm18, zmm19, zmm20, zmm21, zmm23, zmm24, zmm26, zmm27,
            zmm29, zmm30;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
           xmm8;

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // LOOP_5x32J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm3)
        ZERO_ACC_ZMM_4_REG( zmm4,  zmm6,  zmm8,  zmm9)
        ZERO_ACC_ZMM_4_REG(zmm10, zmm11, zmm12, zmm13)
        ZERO_ACC_ZMM_4_REG(zmm14, zmm15, zmm16, zmm17)
        ZERO_ACC_ZMM_4_REG(zmm18, zmm19, zmm20, zmm21)
        ZERO_ACC_ZMM_4_REG(zmm23, zmm24, zmm26, zmm27)
        ZERO_ACC_ZMM_2_REG(zmm29, zmm30)

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_4_REG(xmm4, xmm5, xmm6, xmm7)
        xmm8 = _mm_setzero_ps();

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);
            zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
            zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
            zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
            zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
            zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm3  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);
            zmm4  = _mm512_maskz_loadu_ps(m_mask, a_temp + 4*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
            zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
            zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
            zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
            zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm21, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm24, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm27, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm30, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm8, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm11);
            xmm2 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm2);
            xmm2 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm7 = _mm_mul_ps(xmm3, xmm0);
            xmm8 = _mm_mul_ps(xmm2, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm21, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm24, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm1);

            ymm2 = _mm512_extractf32x8_ps(zmm27, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm30, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm8, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm11);
            xmm2 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm2);
            xmm2 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm3 = _mm_mul_ps(xmm3, xmm0);
            xmm2 = _mm_mul_ps(xmm2, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
            xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
            xmm1 = _mm_loadu_ps(c_temp + 4*rs_c0);
            xmm8 = _mm_fmadd_ps(xmm0, xmm1, xmm2);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x32F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_5x32F:
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

POST_OPS_RELU_SCALE_5x32F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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

POST_OPS_GELU_TANH_5x32F:
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

POST_OPS_GELU_ERF_5x32F:
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

POST_OPS_CLIP_5x32F:
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

POST_OPS_DOWNSCALE_5x32F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

                //c[3, 0-3]
                F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);

                //c[4, 0-3]
                F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
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
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 4 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,4)
                    }
                    else
                    {
                        zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                                                    post_ops_attr.post_op_c_i + 4 ) );
                    }
                }

                //c[4, 0-3]
                F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_MATRIX_ADD_5x32F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 3 ) );
                    scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_5x32F:
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

POST_OPS_SWISH_5x32F:
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

POST_OPS_TANH_5x32F:
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

POST_OPS_SIGMOID_5x32F:
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

POST_OPS_5x32F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm7);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm8);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp = c_temp + 6*rs_c0;
        a_temp = a_temp + 6*rs_a0;
    }   // END LOOP_5x32J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x32_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm6,  zmm8,  zmm9, zmm10,
           zmm11, zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18,
           zmm19, zmm20, zmm23, zmm26, zmm29;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // LOOP_4x32J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm3)
        ZERO_ACC_ZMM_4_REG( zmm6,  zmm8,  zmm9, zmm10)
        ZERO_ACC_ZMM_4_REG(zmm11, zmm12, zmm13, zmm14)
        ZERO_ACC_ZMM_4_REG(zmm15, zmm16, zmm17, zmm18)
        ZERO_ACC_ZMM_4_REG(zmm19, zmm20, zmm23, zmm26)
        zmm29 = _mm512_setzero_ps();

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_4_REG(xmm4, xmm5, xmm6, xmm7)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm3  = _mm512_loadu_ps(a_temp + 3*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm3  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
            zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
            zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
            zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
            zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm7 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm7 = _mm_mul_ps(xmm7, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);

            ymm0 = _mm512_extractf32x8_ps(zmm20, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm20), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm23, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm29, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm5);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm6, ymm7);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm3 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm3 = _mm_mul_ps(xmm3, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
            xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x32F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_4x32F:
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

POST_OPS_RELU_SCALE_4x32F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
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

POST_OPS_GELU_TANH_4x32F:
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

POST_OPS_GELU_ERF_4x32F:
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

POST_OPS_CLIP_4x32F:
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

POST_OPS_DOWNSCALE_4x32F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

                //c[3, 0-3]
                F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
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
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_MATRIX_ADD_4x32F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_4x32F:
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

POST_OPS_SWISH_4x32F:
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

POST_OPS_TANH_4x32F:
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

POST_OPS_SIGMOID_4x32F:
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

POST_OPS_4x32F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm7);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_4x32J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x32_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm2,  zmm6,  zmm8, zmm9, zmm10, zmm11,
           zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18, zmm19;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // LOOP_3x32J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm2,  zmm6)
        ZERO_ACC_ZMM_4_REG( zmm8,  zmm9, zmm10, zmm11)
        ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15)
        ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19)

        // zero out all ymm registers
        ZERO_YMM_ALL

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_3_REG(xmm4, xmm5, xmm6)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm2  = _mm512_loadu_ps(a_temp + 2*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm2  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
            zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
            zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
            zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
            zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm10, 1);
            ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm10), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm13, 1);
            ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm13), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm16, 1);
            ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm16), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm19, 1);
            ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm19), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm6, ymm9);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm12, ymm15);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm6 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
            xmm6 = _mm_mul_ps(xmm6, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
            xmm1 = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm6 = _mm_fmadd_ps(xmm0, xmm1, xmm6);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x32F:
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
                if ( post_ops_list_temp->stor_type == BF16 )
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

POST_OPS_RELU_3x32F:
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

POST_OPS_RELU_SCALE_3x32F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            // c[2,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_3x32F:
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

POST_OPS_GELU_ERF_3x32F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            // c[2,0-3]
            GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_3x32F:
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

POST_OPS_DOWNSCALE_3x32F:
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
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            if ( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if ( is_bf16 == TRUE )
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

                //c[2, 0-3]
                F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);
            }
            else
            {
                if ( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 0 ) );
                    selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 1 ) );
                    selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                post_ops_attr.post_op_c_i + 2 ) );
                }
                if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_3x32F:
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
                scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_3x32F:
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

POST_OPS_SWISH_3x32F:
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

POST_OPS_TANH_3x32F:
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

POST_OPS_SIGMOID_3x32F:
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

POST_OPS_3x32F_DISABLE:
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
            _mm_storeu_ps(c_temp, xmm4);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm5);
            c_temp += rs_c;
            _mm_storeu_ps(c_temp, xmm6);
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_3x32J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x32_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512  zmm0,  zmm1,  zmm6, zmm8, zmm9, zmm11, zmm12, zmm14,
           zmm15, zmm17, zmm18;

    __m256  ymm0,  ymm1,  ymm2,  ymm3, ymm4, ymm5, ymm7, ymm8,
           ymm10, ymm11, ymm13, ymm14;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // LOOP_2x32J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm1,  zmm6,  zmm8)
        ZERO_ACC_ZMM_4_REG( zmm9, zmm11, zmm12, zmm14)
        ZERO_ACC_ZMM_3_REG(zmm15, zmm17, zmm18)

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG( ymm0,  ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG( ymm4,  ymm5,  ymm7,  ymm8)
        ZERO_ACC_YMM_4_REG(ymm10, ymm11, ymm13, ymm14)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        ZERO_ACC_XMM_2_REG(xmm4, xmm5)

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm1  = _mm512_loadu_ps(a_temp + 1*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm1  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
            zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
            zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
            zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
            zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
        }

        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm9, 1);
            ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm9), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm12, 1);
            ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm12), ymm3);

            ymm0 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm15, 1);
            ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm15), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm18, 1);
            ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm18), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            ymm0 = _mm256_hadd_ps(ymm5, ymm8);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm11, ymm14);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm5 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
            xmm5 = _mm_mul_ps(xmm5, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
            xmm1 = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm5 = _mm_fmadd_ps(xmm0, xmm1, xmm5);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x32F:
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

POST_OPS_RELU_2x32F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            // c[1,0-3]
            xmm5 = _mm_max_ps( xmm5, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_2x32F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_2x32F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            // c[1,0-3]
            GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_2x32F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            // c[1,0-3]
            GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_2x32F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            // c[1,0-3]
            CLIP_F32S_SSE(xmm5, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_2x32F:
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

POST_OPS_MATRIX_ADD_2x32F:
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

POST_OPS_MATRIX_MUL_2x32F:
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

POST_OPS_SWISH_2x32F:
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

POST_OPS_TANH_2x32F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            // c[1,0-3]
            TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_2x32F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            // c[1,0-3]
            SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_2x32F_DISABLE:
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

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_2x32J
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x32_rd)
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

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    float *abuf = (float* )a;
    float *bbuf = (float* )b;
    float *cbuf = (float* )c;

    __m512 zmm0, zmm6, zmm8, zmm11, zmm14, zmm17;

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm7, ymm10, ymm13;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // LOOP_1x32J
    {
        // Reset temporary head to base of post_ops_list.
        lpgemm_post_op* post_ops_list_temp = post_ops_list;

        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        ZERO_ACC_ZMM_4_REG( zmm0,  zmm6, zmm8, zmm11)
        ZERO_ACC_ZMM_2_REG(zmm14, zmm17)

        // zero out all ymm registers
        ZERO_ACC_YMM_4_REG(ymm0, ymm1,  ymm2,  ymm3)
        ZERO_ACC_YMM_4_REG(ymm4, ymm7, ymm10, ymm13)

        // zero out all xmm registers
        ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3)
        xmm4 = _mm_setzero_ps();

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm0  = _mm512_loadu_ps(a_temp + 0*rs_a0);

            zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

            zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

            zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

            zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm0  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);

            zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
        }


        if ( beta == 0 )
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);
        }
        else
        {
            ymm0 = _mm512_extractf32x8_ps(zmm8, 1);
            ymm4 = _mm256_add_ps(_mm512_castps512_ps256(zmm8), ymm0);
            ymm1 = _mm512_extractf32x8_ps(zmm11, 1);
            ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm11), ymm1);
            ymm2 = _mm512_extractf32x8_ps(zmm14, 1);
            ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm14), ymm2);
            ymm3 = _mm512_extractf32x8_ps(zmm17, 1);
            ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm17), ymm3);

            ymm0 = _mm256_hadd_ps(ymm4, ymm7);
            xmm1 = _mm256_extractf128_ps(ymm0, 1);
            xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
            ymm2 = _mm256_hadd_ps(ymm10, ymm13);
            xmm3 = _mm256_extractf128_ps(ymm2, 1);
            xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
            xmm4 = _mm_hadd_ps(xmm0, xmm2);

            // ALPHA SCAL
            xmm0 = _mm_broadcast_ss(&alpha);
            xmm4 = _mm_mul_ps(xmm4, xmm0);

            // BETA SCAL
            xmm0 = _mm_broadcast_ss(&beta);
            xmm1 = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        }

        // Post Ops
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x64F:
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

POST_OPS_RELU_1x64F:
        {
            xmm0 = _mm_setzero_ps();

            // c[0,0-3]
            xmm4 = _mm_max_ps( xmm4, xmm0 );

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_1x64F:
        {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_setzero_ps();

            // c[0,0-3]
            RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_TANH_1x64F:
        {
            __m128 dn, x_tanh;
            __m128i q;

            // c[0,0-3]
            GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_GELU_ERF_1x64F:
        {
            // c[0,0-3]
            GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_CLIP_1x64F:
        {
            xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            // c[0,0-3]
            CLIP_F32S_SSE(xmm4, xmm0, xmm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_DOWNSCALE_1x64F:
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

POST_OPS_MATRIX_ADD_1x64F:
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

POST_OPS_MATRIX_MUL_1x64F:
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

POST_OPS_SWISH_1x64F:
        {
            xmm0 =
                _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_TANH_1x64F:
        {
            __m128 dn;
            __m128i q;

            // c[0,0-3]
            TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_SIGMOID_1x64F:
        {
            __m128 z, dn;
            __m128i ex_out;

            // c[0,0-3]
            SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_1x64F_DISABLE:
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
            c_temp += rs_c;
        }

        post_ops_attr.post_op_c_j += 4;

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }   // END LOOP_1x32J
}

#endif
