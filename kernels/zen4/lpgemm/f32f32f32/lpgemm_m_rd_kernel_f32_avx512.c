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

#define MR 6
#define NR 64

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_6x64m_rd)
{
    uint64_t n_left = n0 % NR;      // n0 is expected to be n0<=NR

    static void* post_ops_labels[] =
    {
        &&POST_OPS_6x64F_DISABLE,
        &&POST_OPS_BIAS_6x64F,
        &&POST_OPS_RELU_6x64F,
        &&POST_OPS_RELU_SCALE_6x64F,
        &&POST_OPS_GELU_TANH_6x64F,
        &&POST_OPS_GELU_ERF_6x64F,
        &&POST_OPS_CLIP_6x64F,
        &&POST_OPS_DOWNSCALE_6x64F,
        &&POST_OPS_MATRIX_ADD_6x64F,
        &&POST_OPS_SWISH_6x64F,
        &&POST_OPS_MATRIX_MUL_6x64F,
        &&POST_OPS_TANH_6x64F,
        &&POST_OPS_SIGMOID_6x64F
    };

    // First check whether this is a edge case in the n dimension.
    // If so, dispatch other 6x?m kernels, as needed.
    if ( n_left )
    {
        float* restrict cij = (float*)c;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;

            lpgemm_rowvar_f32f32f32of32_avx512_6x48m_rd
            (
              m0, nr_cur, k0,
              ai, rs_a, cs_a, ps_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c;
            bj  += nr_cur*cs_b;
            n_left -= nr_cur;
            post_ops_attr.post_op_c_j += nr_cur;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;

            lpgemm_rowvar_f32f32f32of32_avx512_6x32m_rd
            (
              m0, nr_cur, k0,
              ai, rs_a, cs_a, ps_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c;
            bj  += nr_cur*cs_b;
            n_left -= nr_cur;
            post_ops_attr.post_op_c_j += nr_cur;
        }

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;

            lpgemm_rowvar_f32f32f32of32_6x16m_rd
            (
              m0, nr_cur, k0,
              ai, rs_a, cs_a, ps_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c;
            bj  += nr_cur*cs_b;
            n_left -= nr_cur;
            post_ops_attr.post_op_c_j += nr_cur;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;

            lpgemm_rowvar_f32f32f32of32_6x8m_rd
            (
              m0, nr_cur, k0,
              ai, rs_a, cs_a, ps_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c;
            bj  += nr_cur*cs_b;
            n_left -= nr_cur;
            post_ops_attr.post_op_c_j += nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;

            lpgemm_rowvar_f32f32f32of32_6x4m_rd
            (
              m0, nr_cur, k0,
              ai, rs_a, cs_a, ps_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c;
            bj  += nr_cur*cs_b;
            n_left -= nr_cur;
            post_ops_attr.post_op_c_j += nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;

            lpgemm_rowvar_f32f32f32of32_6x2m_rd
            (
              m0, nr_cur, k0,
              ai, rs_a, cs_a, ps_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c;
            bj  += nr_cur*cs_b;
            n_left -= nr_cur;
            post_ops_attr.post_op_c_j += nr_cur;
        }

        if ( 1 == n_left )
        {
            const dim_t nr_cur = 1;

            lpgemm_rowvar_f32f32f32of32_6x1m_rd
            (
              m0, nr_cur, k0,
              ai, rs_a, cs_a, ps_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta,
              post_ops_list, post_ops_attr
            );
        }

        return;
    }

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a0 = rs_a;
    uint64_t cs_b0 = cs_b;
    uint64_t rs_c0 = rs_c;
    uint64_t cs_c0 = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm4,  zmm5,  zmm6,  zmm8,
            zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15, zmm16,
           zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24,
           zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
           xmm8, xmm9;

    // Save c_j index for restoring later.
    uint64_t post_op_c_j_save = post_ops_attr.post_op_c_j;

    dim_t jj, ii;
    for ( jj = 0; jj < 64; jj += 4 )    // LOOP_6x64J
    {
        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // LOOP_6x4I
        {
            // Reset temporary head to base of post_ops_list.
            lpgemm_post_op* post_ops_list_temp = post_ops_list;

            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all zmm registers
            ZERO_ACC_ZMM_4_REG(zmm0, zmm1, zmm2, zmm3)
            ZERO_ACC_ZMM_4_REG(zmm4, zmm5, zmm6, zmm8)
            ZERO_ACC_ZMM_4_REG(zmm9, zmm10, zmm11, zmm12)
            ZERO_ACC_ZMM_4_REG(zmm13, zmm14, zmm15, zmm16)
            ZERO_ACC_ZMM_4_REG(zmm17, zmm18, zmm19, zmm20)
            ZERO_ACC_ZMM_4_REG(zmm21, zmm22, zmm23, zmm24)
            ZERO_ACC_ZMM_4_REG(zmm25, zmm26, zmm27, zmm28)
            ZERO_ACC_ZMM_3_REG(zmm29, zmm30, zmm31)

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
                    zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);
                    zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                    zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                    zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                    zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                    zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                    zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                    zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                    zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                    zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                    zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                    zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                    zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                    zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                    zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                    zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                    zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                    zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                    zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                    zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                    zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                    zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                    zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                    zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                    zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                    zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                    zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                    zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                    zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                    zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                    zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                    zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                    zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                    zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                    zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                    zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                    zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                    zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                    zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                    zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                    zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                    zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                    zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                    zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                    zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                    zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                    zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                    zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                    zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                    zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                    zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                    zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                    zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                    zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                    zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                    zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                    zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                    zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                    zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                zmm5  = _mm512_maskz_loadu_ps(m_mask, a_temp + 5*rs_a0);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);
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
                ymm1 = _mm512_extractf32x8_ps(zmm21, 1);
                ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm22, 1);
                ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm22), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm23, 1);
                ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm24, 1);
                ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm25, 1);
                ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm25), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
                ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm27, 1);
                ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm28, 1);
                ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm28), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm29, 1);
                ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm30, 1);
                ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm31, 1);
                ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm31), ymm3);

                ymm0 = _mm256_hadd_ps(ymm4, ymm7);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm10, ymm13);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm7 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm5, ymm8);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm11, ymm14);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm3 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm6, ymm9);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm12, ymm15);
                xmm1 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);
                xmm2 = _mm_hadd_ps(xmm0, xmm2);

                // ALPHA SCAL
                xmm0 = _mm_broadcast_ss(&alpha);
                xmm7 = _mm_mul_ps(xmm7, xmm0);
                xmm8 = _mm_mul_ps(xmm3, xmm0);
                xmm9 = _mm_mul_ps(xmm2, xmm0);
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
                ymm1 = _mm512_extractf32x8_ps(zmm21, 1);
                ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm22, 1);
                ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm22), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm23, 1);
                ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm24, 1);
                ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm25, 1);
                ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm25), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
                ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm27, 1);
                ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm28, 1);
                ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm28), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm29, 1);
                ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm30, 1);
                ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm31, 1);
                ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm31), ymm3);

                ymm0 = _mm256_hadd_ps(ymm4, ymm7);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm10, ymm13);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm7 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm5, ymm8);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm11, ymm14);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm3 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm6, ymm9);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm12, ymm15);
                xmm1 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);
                xmm2 = _mm_hadd_ps(xmm0, xmm2);

                // ALPHA SCAL
                xmm0 = _mm_broadcast_ss(&alpha);
                xmm7 = _mm_mul_ps(xmm7, xmm0);
                xmm3 = _mm_mul_ps(xmm3, xmm0);
                xmm2 = _mm_mul_ps(xmm2, xmm0);

                // BETA SCAL
                xmm0 = _mm_broadcast_ss(&beta);
                xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
                xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
                xmm1 = _mm_loadu_ps(c_temp + 4*rs_c0);
                xmm8 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
                xmm1 = _mm_loadu_ps(c_temp + 5*rs_c0);
                xmm9 = _mm_fmadd_ps(xmm0, xmm1, xmm2);
            }

            // Post Ops
            POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x64F:
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

POST_OPS_RELU_6x64F:
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

POST_OPS_RELU_SCALE_6x64F:
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

                // c[5,0-3]
                RELU_SCALE_OP_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

                POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
            }

POST_OPS_GELU_TANH_6x64F:
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

POST_OPS_GELU_ERF_6x64F:
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

POST_OPS_CLIP_6x64F:
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

POST_OPS_DOWNSCALE_6x64F:
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

                    //c[5, 0-3]
                    F32_SCL_MULRND_SSE(xmm9, selector0, zero_point0);
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
                        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                    post_ops_attr.post_op_c_i + 5 ) );
                    }
                    if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                    {
                        if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_6x64F:
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr6 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                        scl_fctr6 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_6x64F:
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

POST_OPS_SWISH_6x64F:
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

POST_OPS_TANH_6x64F:
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

POST_OPS_SIGMOID_6x64F:
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

POST_OPS_6x64F_DISABLE:
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
                _mm_storeu_ps(c_temp, xmm4);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm5);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm6);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm7);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm8);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm9);
            }

            post_ops_attr.post_op_c_i += 6;

            cbuf = cbuf + 6*rs_c0;
            abuf = abuf + 6*rs_a0;
        }   // END LOOP_6x4I

        post_ops_attr.post_op_c_j += 4;
        post_ops_attr.post_op_c_i  = 0;
    }   // END LOOP_6x64J

    // Reset the value of post_op_c_j to point to the beginning.
    post_ops_attr.post_op_c_j = post_op_c_j_save;

    // Update the post_op_c_i value to account for the number of rows.
    post_ops_attr.post_op_c_i = MR * m_iter;

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 5 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_5x64_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 4 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_4x64_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 3 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_3x64_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 2 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_2x64_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_1x64_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_6x48m_rd)
{
    static void* post_ops_labels[] =
    {
        &&POST_OPS_6x48F_DISABLE,
        &&POST_OPS_BIAS_6x48F,
        &&POST_OPS_RELU_6x48F,
        &&POST_OPS_RELU_SCALE_6x48F,
        &&POST_OPS_GELU_TANH_6x48F,
        &&POST_OPS_GELU_ERF_6x48F,
        &&POST_OPS_CLIP_6x48F,
        &&POST_OPS_DOWNSCALE_6x48F,
        &&POST_OPS_MATRIX_ADD_6x48F,
        &&POST_OPS_SWISH_6x48F,
        &&POST_OPS_MATRIX_MUL_6x48F,
        &&POST_OPS_TANH_6x48F,
        &&POST_OPS_SIGMOID_6x48F
    };

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm4,  zmm5,  zmm6,  zmm8,
            zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15, zmm16,
           zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24,
           zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
           xmm8, xmm9;

    // Save c_j index for restoring later.
    uint64_t post_op_c_j_save = post_ops_attr.post_op_c_j;

    dim_t jj, ii;
    for ( jj = 0; jj < 48; jj += 4 )    // LOOP_6x48J
    {
        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // LOOP_6x4I
        {
            // Reset temporary head to base of post_ops_list.
            lpgemm_post_op* post_ops_list_temp = post_ops_list;

            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all zmm registers
            ZERO_ACC_ZMM_4_REG(zmm0, zmm1, zmm2, zmm3)
            ZERO_ACC_ZMM_4_REG(zmm4, zmm5, zmm6, zmm8)
            ZERO_ACC_ZMM_4_REG(zmm9, zmm10, zmm11, zmm12)
            ZERO_ACC_ZMM_4_REG(zmm13, zmm14, zmm15, zmm16)
            ZERO_ACC_ZMM_4_REG(zmm17, zmm18, zmm19, zmm20)
            ZERO_ACC_ZMM_4_REG(zmm21, zmm22, zmm23, zmm24)
            ZERO_ACC_ZMM_4_REG(zmm25, zmm26, zmm27, zmm28)
            ZERO_ACC_ZMM_3_REG(zmm29, zmm30, zmm31)

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
                    zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);
                    zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                    zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                    zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                    zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                    zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                    zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                    zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                    zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                    zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                    zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                    zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                    zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                    zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                    zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                    zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                    zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                    zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                    zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                    zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                    zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                    zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                    zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                    zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                    zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                    zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                    zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                    zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                    zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                    zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                    zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                    zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                    zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                    zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                    zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                    zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                    zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                    zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                    zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                    zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                    zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                    zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                    zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                    zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                    zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                    zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                    zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                    zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                    zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                    zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                    zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                    zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                    zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                    zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                    zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                    zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                    zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                    zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                    zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                zmm5  = _mm512_maskz_loadu_ps(m_mask, a_temp + 5*rs_a0);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);
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
                ymm1 = _mm512_extractf32x8_ps(zmm21, 1);
                ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm22, 1);
                ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm22), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm23, 1);
                ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm24, 1);
                ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm25, 1);
                ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm25), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
                ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm27, 1);
                ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm28, 1);
                ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm28), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm29, 1);
                ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm30, 1);
                ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm31, 1);
                ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm31), ymm3);

                ymm0 = _mm256_hadd_ps(ymm4, ymm7);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm10, ymm13);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm7 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm5, ymm8);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm11, ymm14);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm3 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm6, ymm9);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm12, ymm15);
                xmm1 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);
                xmm2 = _mm_hadd_ps(xmm0, xmm2);

                // ALPHA SCAL
                xmm0 = _mm_broadcast_ss(&alpha);
                xmm7 = _mm_mul_ps(xmm7, xmm0);
                xmm8 = _mm_mul_ps(xmm3, xmm0);
                xmm9 = _mm_mul_ps(xmm2, xmm0);
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
                ymm1 = _mm512_extractf32x8_ps(zmm21, 1);
                ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm22, 1);
                ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm22), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm23, 1);
                ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm24, 1);
                ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm25, 1);
                ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm25), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
                ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm27, 1);
                ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm28, 1);
                ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm28), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm29, 1);
                ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm30, 1);
                ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm31, 1);
                ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm31), ymm3);

                ymm0 = _mm256_hadd_ps(ymm4, ymm7);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm10, ymm13);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm7 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm5, ymm8);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm11, ymm14);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm3 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm6, ymm9);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm12, ymm15);
                xmm1 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);
                xmm2 = _mm_hadd_ps(xmm0, xmm2);

                // ALPHA SCAL
                xmm0 = _mm_broadcast_ss(&alpha);
                xmm7 = _mm_mul_ps(xmm7, xmm0);
                xmm3 = _mm_mul_ps(xmm3, xmm0);
                xmm2 = _mm_mul_ps(xmm2, xmm0);

                // BETA SCAL
                xmm0 = _mm_broadcast_ss(&beta);
                xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
                xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
                xmm1 = _mm_loadu_ps(c_temp + 4*rs_c0);
                xmm8 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
                xmm1 = _mm_loadu_ps(c_temp + 5*rs_c0);
                xmm9 = _mm_fmadd_ps(xmm0, xmm1, xmm2);
            }

            // Post Ops
            POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x48F:
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

POST_OPS_RELU_6x48F:
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

POST_OPS_RELU_SCALE_6x48F:
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

                // c[5,0-3]
                RELU_SCALE_OP_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

                POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
            }

POST_OPS_GELU_TANH_6x48F:
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

POST_OPS_GELU_ERF_6x48F:
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

POST_OPS_CLIP_6x48F:
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

POST_OPS_DOWNSCALE_6x48F:
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

                    //c[5, 0-3]
                    F32_SCL_MULRND_SSE(xmm9, selector0, zero_point0);
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
                        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                    post_ops_attr.post_op_c_i + 5 ) );
                    }

                    if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                    {
                        if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_6x48F:
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr6 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                        scl_fctr6 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_6x48F:
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

POST_OPS_SWISH_6x48F:
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

POST_OPS_TANH_6x48F:
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

POST_OPS_SIGMOID_6x48F:
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

POST_OPS_6x48F_DISABLE:
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
                _mm_storeu_ps(c_temp, xmm4);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm5);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm6);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm7);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm8);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm9);
            }

            post_ops_attr.post_op_c_i += 6;

            cbuf = cbuf + 6*rs_c0;
            abuf = abuf + 6*rs_a0;
        }   // END LOOP_6x4I

        post_ops_attr.post_op_c_j += 4;
        post_ops_attr.post_op_c_i  = 0;
    }   // END LOOP_6x48J

    // Reset the value of post_op_c_j to point to the beginning.
    post_ops_attr.post_op_c_j = post_op_c_j_save;

    // Update the post_op_c_i value to account for the number of rows.
    post_ops_attr.post_op_c_i = MR * m_iter;

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 5 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_5x48_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 4 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_4x48_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 3 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_3x48_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 2 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_2x48_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_1x48_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_6x32m_rd)
{
    static void* post_ops_labels[] =
    {
        &&POST_OPS_6x32F_DISABLE,
        &&POST_OPS_BIAS_6x32F,
        &&POST_OPS_RELU_6x32F,
        &&POST_OPS_RELU_SCALE_6x32F,
        &&POST_OPS_GELU_TANH_6x32F,
        &&POST_OPS_GELU_ERF_6x32F,
        &&POST_OPS_CLIP_6x32F,
        &&POST_OPS_DOWNSCALE_6x32F,
        &&POST_OPS_MATRIX_ADD_6x32F,
        &&POST_OPS_SWISH_6x32F,
        &&POST_OPS_MATRIX_MUL_6x32F,
        &&POST_OPS_TANH_6x32F,
        &&POST_OPS_SIGMOID_6x32F
    };

    uint64_t k_iter64 = k0 / 64;
    uint64_t k_left64 = k0 % 64;
    uint64_t k_iter32 = k_left64 / 32;
    uint64_t k_left32 = k_left64 % 32;
    uint64_t k_iter16 = k_left32 / 16;
    uint64_t k_left16 = k_left32 % 16;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m512  zmm0,  zmm1,  zmm2,  zmm3,  zmm4,  zmm5,  zmm6,  zmm8,
            zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15, zmm16,
           zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24,
           zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    __m256 ymm0, ymm1,  ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
           ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
           xmm8, xmm9;

    // Save c_j index for restoring later.
    uint64_t post_op_c_j_save = post_ops_attr.post_op_c_j;

    dim_t jj, ii;
    for ( jj = 0; jj < 32; jj += 4 )    // LOOP_6x32J
    {
        float *abuf = (float* )a;
        float *bbuf = (float* )b;
        float *cbuf = (float* )c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // LOOP_6x4I
        {
            // Reset temporary head to base of post_ops_list.
            lpgemm_post_op* post_ops_list_temp = post_ops_list;

            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all zmm registers
            ZERO_ACC_ZMM_4_REG(zmm0, zmm1, zmm2, zmm3)
            ZERO_ACC_ZMM_4_REG(zmm4, zmm5, zmm6, zmm8)
            ZERO_ACC_ZMM_4_REG(zmm9, zmm10, zmm11, zmm12)
            ZERO_ACC_ZMM_4_REG(zmm13, zmm14, zmm15, zmm16)
            ZERO_ACC_ZMM_4_REG(zmm17, zmm18, zmm19, zmm20)
            ZERO_ACC_ZMM_4_REG(zmm21, zmm22, zmm23, zmm24)
            ZERO_ACC_ZMM_4_REG(zmm25, zmm26, zmm27, zmm28)
            ZERO_ACC_ZMM_3_REG(zmm29, zmm30, zmm31)

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
                    zmm4  = _mm512_loadu_ps(a_temp + 4*rs_a0);
                    zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                    zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                    zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                    zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                    zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                    zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                    zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                    zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                    zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                    zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                    zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                    zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                    zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                    zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                    zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                    zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                    zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                    zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                    zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                    zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                    zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                    zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                    zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                    zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                    zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                    zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                    zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                    zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                    zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                    zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                    zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                    zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                    zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                    zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                    zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                    zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                    zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                    zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                    zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                    zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                    zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                    zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                    zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                    zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                    zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                    zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                    zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                    zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                    zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                    zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                    zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                    zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                    zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                    zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                    zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                    zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                    zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                    zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                zmm5  = _mm512_loadu_ps(a_temp + 5*rs_a0);

                zmm6  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                zmm6  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                zmm6  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                zmm6  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);

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
                zmm5  = _mm512_maskz_loadu_ps(m_mask, a_temp + 5*rs_a0);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

                zmm8  = _mm512_fmadd_ps(zmm0, zmm6, zmm8);
                zmm9  = _mm512_fmadd_ps(zmm1, zmm6, zmm9);
                zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10);
                zmm20 = _mm512_fmadd_ps(zmm3, zmm6, zmm20);
                zmm21 = _mm512_fmadd_ps(zmm4, zmm6, zmm21);
                zmm22 = _mm512_fmadd_ps(zmm5, zmm6, zmm22);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

                zmm11 = _mm512_fmadd_ps(zmm0, zmm6, zmm11);
                zmm12 = _mm512_fmadd_ps(zmm1, zmm6, zmm12);
                zmm13 = _mm512_fmadd_ps(zmm2, zmm6, zmm13);
                zmm23 = _mm512_fmadd_ps(zmm3, zmm6, zmm23);
                zmm24 = _mm512_fmadd_ps(zmm4, zmm6, zmm24);
                zmm25 = _mm512_fmadd_ps(zmm5, zmm6, zmm25);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

                zmm14 = _mm512_fmadd_ps(zmm0, zmm6, zmm14);
                zmm15 = _mm512_fmadd_ps(zmm1, zmm6, zmm15);
                zmm16 = _mm512_fmadd_ps(zmm2, zmm6, zmm16);
                zmm26 = _mm512_fmadd_ps(zmm3, zmm6, zmm26);
                zmm27 = _mm512_fmadd_ps(zmm4, zmm6, zmm27);
                zmm28 = _mm512_fmadd_ps(zmm5, zmm6, zmm28);

                zmm6  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

                zmm17 = _mm512_fmadd_ps(zmm0, zmm6, zmm17);
                zmm18 = _mm512_fmadd_ps(zmm1, zmm6, zmm18);
                zmm19 = _mm512_fmadd_ps(zmm2, zmm6, zmm19);
                zmm29 = _mm512_fmadd_ps(zmm3, zmm6, zmm29);
                zmm30 = _mm512_fmadd_ps(zmm4, zmm6, zmm30);
                zmm31 = _mm512_fmadd_ps(zmm5, zmm6, zmm31);
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
                ymm1 = _mm512_extractf32x8_ps(zmm21, 1);
                ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm22, 1);
                ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm22), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm23, 1);
                ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm24, 1);
                ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm25, 1);
                ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm25), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
                ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm27, 1);
                ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm28, 1);
                ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm28), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm29, 1);
                ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm30, 1);
                ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm31, 1);
                ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm31), ymm3);

                ymm0 = _mm256_hadd_ps(ymm4, ymm7);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm10, ymm13);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm7 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm5, ymm8);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm11, ymm14);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm3 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm6, ymm9);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm12, ymm15);
                xmm1 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);
                xmm2 = _mm_hadd_ps(xmm0, xmm2);

                // ALPHA SCAL
                xmm0 = _mm_broadcast_ss(&alpha);
                xmm7 = _mm_mul_ps(xmm7, xmm0);
                xmm8 = _mm_mul_ps(xmm3, xmm0);
                xmm9 = _mm_mul_ps(xmm2, xmm0);
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
                ymm1 = _mm512_extractf32x8_ps(zmm21, 1);
                ymm5 = _mm256_add_ps(_mm512_castps512_ps256(zmm21), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm22, 1);
                ymm6 = _mm256_add_ps(_mm512_castps512_ps256(zmm22), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm23, 1);
                ymm7 = _mm256_add_ps(_mm512_castps512_ps256(zmm23), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm24, 1);
                ymm8 = _mm256_add_ps(_mm512_castps512_ps256(zmm24), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm25, 1);
                ymm9 = _mm256_add_ps(_mm512_castps512_ps256(zmm25), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm26, 1);
                ymm10 = _mm256_add_ps(_mm512_castps512_ps256(zmm26), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm27, 1);
                ymm11 = _mm256_add_ps(_mm512_castps512_ps256(zmm27), ymm3);

                ymm0 = _mm512_extractf32x8_ps(zmm28, 1);
                ymm12 = _mm256_add_ps(_mm512_castps512_ps256(zmm28), ymm0);
                ymm1 = _mm512_extractf32x8_ps(zmm29, 1);
                ymm13 = _mm256_add_ps(_mm512_castps512_ps256(zmm29), ymm1);

                ymm2 = _mm512_extractf32x8_ps(zmm30, 1);
                ymm14 = _mm256_add_ps(_mm512_castps512_ps256(zmm30), ymm2);
                ymm3 = _mm512_extractf32x8_ps(zmm31, 1);
                ymm15 = _mm256_add_ps(_mm512_castps512_ps256(zmm31), ymm3);

                ymm0 = _mm256_hadd_ps(ymm4, ymm7);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm10, ymm13);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm7 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm5, ymm8);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm11, ymm14);
                xmm3 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm3);
                xmm3 = _mm_hadd_ps(xmm0, xmm2);

                ymm0 = _mm256_hadd_ps(ymm6, ymm9);
                xmm1 = _mm256_extractf128_ps(ymm0, 1);
                xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm1);
                ymm2 = _mm256_hadd_ps(ymm12, ymm15);
                xmm1 = _mm256_extractf128_ps(ymm2, 1);
                xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm1);
                xmm2 = _mm_hadd_ps(xmm0, xmm2);

                // ALPHA SCAL
                xmm0 = _mm_broadcast_ss(&alpha);
                xmm7 = _mm_mul_ps(xmm7, xmm0);
                xmm3 = _mm_mul_ps(xmm3, xmm0);
                xmm2 = _mm_mul_ps(xmm2, xmm0);

                // BETA SCAL
                xmm0 = _mm_broadcast_ss(&beta);
                xmm1 = _mm_loadu_ps(c_temp + 3*rs_c0);
                xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
                xmm1 = _mm_loadu_ps(c_temp + 4*rs_c0);
                xmm8 = _mm_fmadd_ps(xmm0, xmm1, xmm3);
                xmm1 = _mm_loadu_ps(c_temp + 5*rs_c0);
                xmm9 = _mm_fmadd_ps(xmm0, xmm1, xmm2);
            }

            // Post Ops
            POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x32F:
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

POST_OPS_RELU_6x32F:
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

POST_OPS_RELU_SCALE_6x32F:
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

                // c[5,0-3]
                RELU_SCALE_OP_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

                POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
            }

POST_OPS_GELU_TANH_6x32F:
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

POST_OPS_GELU_ERF_6x32F:
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

POST_OPS_CLIP_6x32F:
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

POST_OPS_DOWNSCALE_6x32F:
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

                    //c[5, 0-3]
                    F32_SCL_MULRND_SSE(xmm9, selector0, zero_point0);
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
                        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                                                    post_ops_attr.post_op_c_i + 5 ) );
                    }
                    if ( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                    {
                        if ( is_bf16 == TRUE )
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

POST_OPS_MATRIX_ADD_6x32F:
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
                    scl_fctr1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr5 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                    scl_fctr6 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
                        scl_fctr6 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
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

POST_OPS_MATRIX_MUL_6x32F:
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

POST_OPS_SWISH_6x32F:
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

POST_OPS_TANH_6x32F:
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

POST_OPS_SIGMOID_6x32F:
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

POST_OPS_6x32F_DISABLE:
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
                _mm_storeu_ps(c_temp, xmm4);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm5);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm6);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm7);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm8);
                c_temp += rs_c;
                _mm_storeu_ps(c_temp, xmm9);
            }

            post_ops_attr.post_op_c_i += 6;

            cbuf = cbuf + 6*rs_c0;
            abuf = abuf + 6*rs_a0;
        }   // END LOOP_6x4I

        post_ops_attr.post_op_c_j += 4;
        post_ops_attr.post_op_c_i  = 0;
    }   // END LOOP_6x32J

    // Reset the value of post_op_c_j to point to the beginning.
    post_ops_attr.post_op_c_j = post_op_c_j_save;

    // Update the post_op_c_i value to account for the number of rows.
    post_ops_attr.post_op_c_i = MR * m_iter;

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 5 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_5x32_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 4 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_4x32_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 3 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_3x32_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 2 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_2x32_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }

        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_avx512_1x32_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}
#endif
