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

#define MR 6
#define NR 16

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x16m_rd)
{
    uint64_t n_left = n0 % 16;

    // First check whether this is a edge case in the n dimension. If so,
    // dispatch other 6x?m kernels, as needed.
    if ( n_left )
    {
        float* restrict cij = (float*)c;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a;

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
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
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
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
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
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
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

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

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

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj, ii;
    for ( jj = 0; jj < 16; jj += 4 )    // SLOOP3X4J
    {
        float *abuf = (float*)a;
        float *bbuf = (float*)b;
        float *cbuf = (float*)c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // SLOOP3X4I
        {
            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all ymm registers
            for ( int i = 0; i < 16; ++i )
            {
                ymm[i] = _mm256_setzero_ps();
            }
            // zero out all xmm registers
            for ( int i = 0; i < 8; ++i )
            {
                xmm[i] = _mm_setzero_ps();
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
            {
                for ( dim_t unroll = 0; unroll < 4; ++unroll )
                {
                    ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                    ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                    ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                    ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                    ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                    ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                    ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                    ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                    ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                    ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                    ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                    ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                    ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                    ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                    ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                    a_temp += 8;
                    b_temp += 8;
                }
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
            {
                ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                a_temp += 8;
                b_temp += 8;
            }

            if ( k_left1 )
            {
                const __m256i mask = masks[k_left1];

                ymm[0] = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
                ymm[1] = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);
                ymm[2] = _mm256_maskload_ps(a_temp + 2*rs_a0, mask);

                ymm[3] = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
                ymm[4] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3] = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                ymm[3]  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
                ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                ymm[3]  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
                ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                a_temp += k_left1;
                b_temp += k_left1;
            }

            // ACCUMULATE
            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCALE
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            if ( beta == 0 )
            {
                xmm[0] = _mm_loadu_ps(c_temp + 0*rs_c0);
                xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
                xmm[2] = _mm_loadu_ps(c_temp + 2*rs_c0);
                _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
                _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
                _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);
                c_temp += 3*rs_c0;
            }
            else    // if ( beta != 0 )
            {
                xmm[0] = _mm_broadcast_ss(&beta);
                xmm[1] = _mm_loadu_ps(c_temp);
                xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
                _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

                xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
                xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
                _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

                xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
                xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
                _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

                c_temp += 3*rs_c0;
            }

            cbuf = cbuf + 3*rs_c0;
            abuf = abuf + 3*rs_a0;
        }
    }

    // -------------------------------------------------------------------------
    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 2 == m_left )
        {


            lpgemm_rowvar_f32f32f32of32_2x16_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
            //cij += mr_cur*rs_c; ai += mr_cur*rs_a; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_1x16_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x8m_rd)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

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

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj, ii;
    for ( jj = 0; jj < 8; jj += 4 )    // SLOOP3X4J
    {
        float *abuf = (float*)a;
        float *bbuf = (float*)b;
        float *cbuf = (float*)c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // SLOOP3X4I
        {
            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all ymm registers
            for ( int i = 0; i < 16; ++i )
            {
                ymm[i] = _mm256_setzero_ps();
            }
            // zero out all xmm registers
            for ( int i = 0; i < 8; ++i )
            {
                xmm[i] = _mm_setzero_ps();
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
            {
                for ( dim_t unroll = 0; unroll < 4; ++unroll )
                {
                    ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                    ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                    ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                    ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                    ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                    ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                    ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                    ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                    ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                    ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                    ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                    ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                    ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                    ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                    ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                    a_temp += 8;
                    b_temp += 8;
                }
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
            {
                ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                a_temp += 8;
                b_temp += 8;
            }

            if ( k_left1 )
            {
                const __m256i mask = masks[k_left1];

                ymm[0] = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
                ymm[1] = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);
                ymm[2] = _mm256_maskload_ps(a_temp + 2*rs_a0, mask);

                ymm[3] = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
                ymm[4] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3] = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                ymm[3]  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
                ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                ymm[3]  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
                ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                a_temp += k_left1;
                b_temp += k_left1;
            }

            // ACCUMULATE
            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCALE
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            if ( beta == 0 )
            {
                xmm[0] = _mm_loadu_ps(c_temp + 0*rs_c0);
                xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
                xmm[2] = _mm_loadu_ps(c_temp + 2*rs_c0);
                _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
                _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
                _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);
                c_temp += 3*rs_c0;
            }
            else    // if ( beta != 0 )
            {
                xmm[0] = _mm_broadcast_ss(&beta);
                xmm[1] = _mm_loadu_ps(c_temp);
                xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
                _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

                xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
                xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
                _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

                xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
                xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
                _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

                c_temp += 3*rs_c0;
            }

            cbuf = cbuf + 3*rs_c0;
            abuf = abuf + 3*rs_a0;
        }
    }

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 2 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_2x8_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
            //cij += mr_cur*rs_c; ai += mr_cur*rs_a; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_1x8_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x4m_rd)
{
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

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

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj, ii;
    for ( jj = 0; jj < 4; jj += 4 )    // SLOOP3X4J
    {
        float *abuf = (float*)a;
        float *bbuf = (float*)b;
        float *cbuf = (float*)c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // SLOOP3X4I
        {
            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all ymm registers
            for ( int i = 0; i < 16; ++i )
            {
                ymm[i] = _mm256_setzero_ps();
            }
            // zero out all xmm registers
            for ( int i = 0; i < 8; ++i )
            {
                xmm[i] = _mm_setzero_ps();
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
            {
                for ( dim_t unroll = 0; unroll < 4; ++unroll )
                {
                    ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                    ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                    ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                    ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                    ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                    ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                    ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                    ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                    ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                    ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                    ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                    ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                    ymm[3]  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                    ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                    ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                    ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                    a_temp += 8;
                    b_temp += 8;
                }
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
            {
                ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 2*cs_b0);
                ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 3*cs_b0);
                ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                a_temp += 8;
                b_temp += 8;
            }

            if ( k_left1 )
            {
                const __m256i mask = masks[k_left1];

                ymm[0] = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
                ymm[1] = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);
                ymm[2] = _mm256_maskload_ps(a_temp + 2*rs_a0, mask);

                ymm[3] = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
                ymm[4] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3] = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                ymm[3]  = _mm256_maskload_ps(b_temp + 2*cs_b0, mask);
                ymm[10] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[10]);
                ymm[11] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[11]);
                ymm[12] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[12]);

                ymm[3]  = _mm256_maskload_ps(b_temp + 3*cs_b0, mask);
                ymm[13] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[13]);
                ymm[14] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[14]);
                ymm[15] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[15]);

                a_temp += k_left1;
                b_temp += k_left1;
            }

            // ACCUMULATE
            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[1] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[1]);

            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCALE
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            if ( beta == 0 )
            {
                xmm[0] = _mm_loadu_ps(c_temp + 0*rs_c0);
                xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
                xmm[2] = _mm_loadu_ps(c_temp + 2*rs_c0);
                _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
                _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
                _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);
                c_temp += 3*rs_c0;
            }
            else    // if ( beta != 0 )
            {
                xmm[0] = _mm_broadcast_ss(&beta);
                xmm[1] = _mm_loadu_ps(c_temp);
                xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
                _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

                xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
                xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
                _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

                xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
                xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
                _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

                c_temp += 3*rs_c0;
            }

            cbuf = cbuf + 3*rs_c0;
            abuf = abuf + 3*rs_a0;
        }
    }

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 2 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_2x4_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
            //cij += mr_cur*rs_c; ai += mr_cur*rs_a; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_1x4_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x2m_rd)
{
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

    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj, ii;
    for ( jj = 0; jj < 4; jj += 4 )    // SLOOP3X4J
    {        
        float *abuf = (float*)a;
        float *bbuf = (float*)b;
        float *cbuf = (float*)c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // SLOOP3X4I
        {
            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all ymm registers
            for ( int i = 0; i < 16; ++i )
            {
                ymm[i] = _mm256_setzero_ps();
            }
            // zero out all xmm registers
            for ( int i = 0; i < 8; ++i )
            {
                xmm[i] = _mm_setzero_ps();
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
            {
                for ( dim_t unroll = 0; unroll < 4; ++unroll )
                {
                    ymm[0] = _mm256_loadu_ps(a_temp + 0*rs_a0);
                    ymm[1] = _mm256_loadu_ps(a_temp + 1*rs_a0);
                    ymm[2] = _mm256_loadu_ps(a_temp + 2*rs_a0);

                    ymm[3] = _mm256_loadu_ps(b_temp + 0*cs_b0);
                    ymm[4] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                    ymm[5] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                    ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                    ymm[3] = _mm256_loadu_ps(b_temp + 1*cs_b0);
                    ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                    ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                    ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                    a_temp += 8;
                    b_temp += 8;
                }
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
            {
                ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3]  = _mm256_loadu_ps(b_temp + 1*cs_b0);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                a_temp += 8;
                b_temp += 8;
            }

            if ( k_left1 )
            {
                const __m256i mask = masks[k_left1];

                ymm[0] = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
                ymm[1] = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);
                ymm[2] = _mm256_maskload_ps(a_temp + 2*rs_a0, mask);

                ymm[3] = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
                ymm[4] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                ymm[3] = _mm256_maskload_ps(b_temp + 1*cs_b0, mask);
                ymm[7] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[7]);
                ymm[8] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[8]);
                ymm[9] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[9]);

                a_temp += k_left1;
                b_temp += k_left1;
            }

            // ACCUMULATE
            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);

            xmm[4] = _mm_hadd_ps(xmm[0], xmm[0]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);

            xmm[5] = _mm_hadd_ps(xmm[0], xmm[0]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);

            xmm[6] = _mm_hadd_ps(xmm[0], xmm[0]);

            // ALPHA SCALE
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            if ( beta == 0 )
            {                xmm[0] = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
                xmm[1] = _mm_maskload_ps(c_temp + 1*rs_c0, m_mask);
                xmm[2] = _mm_maskload_ps(c_temp + 2*rs_c0, m_mask);
                _mm_maskstore_ps(c_temp + 0*rs_c0, m_mask, xmm[4]);
                _mm_maskstore_ps(c_temp + 1*rs_c0, m_mask, xmm[5]);
                _mm_maskstore_ps(c_temp + 2*rs_c0, m_mask, xmm[6]);
            }
            else    // if ( beta != 0 )
            {                xmm[3] = _mm_broadcast_ss(&beta);

                xmm[0] = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
                xmm[0] = _mm_fmadd_ps(xmm[3], xmm[0], xmm[4]);
                _mm_maskstore_ps(c_temp + 0*rs_c0, m_mask, xmm[0]);

                xmm[0] = _mm_maskload_ps(c_temp + 1*rs_c0, m_mask);
                xmm[0] = _mm_fmadd_ps(xmm[3], xmm[0], xmm[5]);
                _mm_maskstore_ps(c_temp + 1*rs_c0, m_mask, xmm[0]);

                xmm[0] = _mm_maskload_ps(c_temp + 2*rs_c0, m_mask);
                xmm[0] = _mm_fmadd_ps(xmm[3], xmm[0], xmm[6]);
                _mm_maskstore_ps(c_temp + 2*rs_c0, m_mask, xmm[0]);
            }

            cbuf = cbuf + 3*rs_c0;
            abuf = abuf + 3*rs_a0;
        }
    }

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 2 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_2x2_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
            //cij += mr_cur*rs_c; ai += mr_cur*rs_a; m_left -= mr_cur;
        }

        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_1x2_rd
            (
              k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
              cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x1m_rd)
{
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

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a0   = rs_a;
    uint64_t cs_b0   = cs_b;
    uint64_t rs_c0   = rs_c;
    uint64_t cs_c0   = cs_c;

    if ( m_iter == 0 ) goto consider_edge_cases;

    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj, ii;
    for ( jj = 0; jj < 4; jj += 4 )    // SLOOP3X4J
    {        
        float *abuf = (float*)a;
        float *bbuf = (float*)b;
        float *cbuf = (float*)c;

        cbuf += jj * cs_c0;
        bbuf += jj * cs_b0;

        for ( ii = 0; ii < m_iter; ++ii )   // SLOOP3X4I
        {
            float* c_temp = cbuf;
            float* a_temp = abuf;
            float* b_temp = bbuf;

            // zero out all ymm registers
            for ( int i = 0; i < 16; ++i )
            {
                ymm[i] = _mm256_setzero_ps();
            }
            // zero out all xmm registers
            for ( int i = 0; i < 8; ++i )
            {
                xmm[i] = _mm_setzero_ps();
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
            {
                for ( dim_t unroll = 0; unroll < 4; ++unroll )
                {
                    ymm[0] = _mm256_loadu_ps(a_temp + 0*rs_a0);
                    ymm[1] = _mm256_loadu_ps(a_temp + 1*rs_a0);
                    ymm[2] = _mm256_loadu_ps(a_temp + 2*rs_a0);

                    ymm[3] = _mm256_loadu_ps(b_temp + 0*cs_b0);
                    ymm[4] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                    ymm[5] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                    ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                    a_temp += 8;
                    b_temp += 8;
                }
            }

            for ( dim_t k_iterator = 0; k_iterator < k_iter8; ++k_iterator )
            {
                ymm[0]  = _mm256_loadu_ps(a_temp + 0*rs_a0);
                ymm[1]  = _mm256_loadu_ps(a_temp + 1*rs_a0);
                ymm[2]  = _mm256_loadu_ps(a_temp + 2*rs_a0);

                ymm[3]  = _mm256_loadu_ps(b_temp + 0*cs_b0);
                ymm[4]  = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5]  = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                a_temp += 8;
                b_temp += 8;
            }

            if ( k_left1 )
            {
                const __m256i mask = masks[k_left1];

                ymm[0] = _mm256_maskload_ps(a_temp + 0*rs_a0, mask);
                ymm[1] = _mm256_maskload_ps(a_temp + 1*rs_a0, mask);
                ymm[2] = _mm256_maskload_ps(a_temp + 2*rs_a0, mask);

                ymm[3] = _mm256_maskload_ps(b_temp + 0*cs_b0, mask);
                ymm[4] = _mm256_fmadd_ps(ymm[0], ymm[3], ymm[4]);
                ymm[5] = _mm256_fmadd_ps(ymm[1], ymm[3], ymm[5]);
                ymm[6] = _mm256_fmadd_ps(ymm[2], ymm[3], ymm[6]);

                a_temp += k_left1;
                b_temp += k_left1;
            }

            // ACCUMULATE
            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[4]);
            ymm[1] = _mm256_hadd_ps(ymm[0], ymm[0]);
            xmm[0] = _mm256_extractf128_ps(ymm[1], 1);
            xmm[4] = _mm_add_ps(_mm256_castps256_ps128(ymm[1]), xmm[0]);
            
            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[5]);
            ymm[1] = _mm256_hadd_ps(ymm[0], ymm[0]);
            xmm[0] = _mm256_extractf128_ps(ymm[1], 1);
            xmm[5] = _mm_add_ps(_mm256_castps256_ps128(ymm[1]), xmm[0]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[6]);
            ymm[1] = _mm256_hadd_ps(ymm[0], ymm[0]);
            xmm[0] = _mm256_extractf128_ps(ymm[1], 1);
            xmm[6] = _mm_add_ps(_mm256_castps256_ps128(ymm[1]), xmm[0]);

            // ALPHA SCALE
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            if ( beta == 0 )
            {
                // printf( "beta == 0\n" );
                xmm[0] = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
                xmm[1] = _mm_maskload_ps(c_temp + 1*rs_c0, m_mask);
                xmm[2] = _mm_maskload_ps(c_temp + 2*rs_c0, m_mask);
                _mm_maskstore_ps(c_temp + 0*rs_c0, m_mask, xmm[4]);
                _mm_maskstore_ps(c_temp + 1*rs_c0, m_mask, xmm[5]);
                _mm_maskstore_ps(c_temp + 2*rs_c0, m_mask, xmm[6]);
            }
            else    // if ( beta != 0 )
            {
                // printf( "beta != 0\n" );
                xmm[3] = _mm_broadcast_ss(&beta);

                xmm[0] = _mm_maskload_ps(c_temp + 0*rs_c0, m_mask);
                xmm[0] = _mm_fmadd_ps(xmm[3], xmm[0], xmm[4]);
                _mm_maskstore_ps(c_temp + 0*rs_c0, m_mask, xmm[0]);

                xmm[0] = _mm_maskload_ps(c_temp + 1*rs_c0, m_mask);
                xmm[0] = _mm_fmadd_ps(xmm[3], xmm[0], xmm[5]);
                _mm_maskstore_ps(c_temp + 1*rs_c0, m_mask, xmm[0]);

                xmm[0] = _mm_maskload_ps(c_temp + 2*rs_c0, m_mask);
                xmm[0] = _mm_fmadd_ps(xmm[3], xmm[0], xmm[6]);
                _mm_maskstore_ps(c_temp + 2*rs_c0, m_mask, xmm[0]);
            }

            cbuf = cbuf + 3*rs_c0;
            abuf = abuf + 3*rs_a0;
        }
    }

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float*)c + i_edge*rs_c0;
        float* restrict bj  = (float*)b;
        float* restrict ai  = (float*)a + i_edge*rs_a0;

        if ( 2 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_2x1_rd
            (
                k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
                cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
            //cij += mr_cur*rs_c; ai += mr_cur*rs_a; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            lpgemm_rowvar_f32f32f32of32_1x1_rd
            (
                k0, ai, rs_a, cs_a, bj, rs_b, cs_b,
                cij, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
            );
        }
    }
}

#endif
