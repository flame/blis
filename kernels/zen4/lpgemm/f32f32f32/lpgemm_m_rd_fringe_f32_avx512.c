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

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x64_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
                zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
                zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
                zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
                zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
                zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
                zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
                zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
                zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
            zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
            zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
            zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
            zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
            zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm[3]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);
            zmm[4]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 4*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
            zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
            zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
            zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
            zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[21], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[21]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[24], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[24]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[27], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[27]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[30], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[30]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[8], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[11]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            c_temp += 2*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[21], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[21]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[24], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[24]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[27], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[27]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[30], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[30]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[8], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[11]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            c_temp += 2*rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x64_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm[3]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            c_temp += rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            c_temp += rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x64_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
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

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x64_rd)
{    
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
        }

        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

            c_temp += 2*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

            c_temp += 2*rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }

}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x64_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 64; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

            c_temp += rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

            c_temp += rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x48_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
                zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
                zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
                zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
                zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
                zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
                zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
                zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
                zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
            zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
            zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
            zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
            zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
            zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm[3]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);
            zmm[4]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 4*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
            zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
            zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
            zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
            zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[21], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[21]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[24], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[24]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[27], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[27]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[30], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[30]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[8], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[11]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            c_temp += 2*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[21], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[21]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[24], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[24]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[27], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[27]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[30], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[30]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[8], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[11]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            c_temp += 2*rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x48_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm[3]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            c_temp += rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            c_temp += rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x48_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
        }

        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
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

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x48_rd)
{    
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
        }

        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

            c_temp += 2*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

            c_temp += 2*rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x48_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 48; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
        }

        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

            c_temp += rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

            c_temp += rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x32_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
                zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
                zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
                zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
                zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
                zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
                zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
                zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
                zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
                zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);
            zmm[4]  = _mm512_loadu_ps(a_temp + 4*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
            zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
            zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
            zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
            zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm[3]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);
            zmm[4]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 4*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);
            zmm[21] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[21]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);
            zmm[24] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[24]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);
            zmm[27] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[27]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
            zmm[30] = _mm512_fmadd_ps(zmm[4], zmm[6], zmm[30]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[21], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[21]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[24], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[24]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[27], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[27]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[30], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[30]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[8], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[11]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            c_temp += 2*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[21], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[21]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[24], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[24]), ymm[1]);

            ymm[2] = _mm512_extractf32x8_ps(zmm[27], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[27]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[30], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[30]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[8], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[11]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            c_temp += 2*rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x32_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
                zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
                zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
                zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
                zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
                zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);
            zmm[3]  = _mm512_loadu_ps(a_temp + 3*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);
            zmm[3]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 3*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);
            zmm[20] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[20]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);
            zmm[23] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[23]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);
            zmm[26] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[26]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
            zmm[29] = _mm512_fmadd_ps(zmm[3], zmm[6], zmm[29]);
        }


        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            c_temp += rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            xmm[1] = _mm_loadu_ps(c_temp + 2*rs_c0);
            xmm[6] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[6]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;

            ymm[0] = _mm512_extractf32x8_ps(zmm[20], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[20]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[23], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[23]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[26], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[26]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[29], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[29]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[5]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[6], ymm[7]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            c_temp += rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x32_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
                zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
                zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
                zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
                zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
                zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);
            zmm[2]  = _mm512_loadu_ps(a_temp + 2*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);
            zmm[2]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 2*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);
            zmm[10] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[10]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);
            zmm[13] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[13]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);
            zmm[16] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[16]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
            zmm[19] = _mm512_fmadd_ps(zmm[2], zmm[6], zmm[19]);
        }

        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);
            _mm_storeu_ps(c_temp + 2*rs_c0, xmm[6]);

            c_temp += 3*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[10], 1);
            ymm[6] = _mm256_add_ps(_mm512_castps512_ps256(zmm[10]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[13], 1);
            ymm[9] = _mm256_add_ps(_mm512_castps512_ps256(zmm[13]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[16], 1);
            ymm[12] = _mm256_add_ps(_mm512_castps512_ps256(zmm[16]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[19], 1);
            ymm[15] = _mm256_add_ps(_mm512_castps512_ps256(zmm[19]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[6], ymm[9]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[12], ymm[15]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[6] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);
            xmm[6] = _mm_mul_ps(xmm[6], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
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

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x32_rd)
{    
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
                zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
                zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
                zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
                zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
                zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);
            zmm[1]  = _mm512_loadu_ps(a_temp + 1*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);
            zmm[1]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 1*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);
            zmm[9]  = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[9]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);
            zmm[12] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[12]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);
            zmm[15] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[15]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
            zmm[18] = _mm512_fmadd_ps(zmm[1], zmm[6], zmm[18]);
        }

        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

            c_temp += 2*rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[9], 1);
            ymm[5] = _mm256_add_ps(_mm512_castps512_ps256(zmm[9]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[12], 1);
            ymm[8] = _mm256_add_ps(_mm512_castps512_ps256(zmm[12]), ymm[3]);

            ymm[0] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[15], 1);
            ymm[11] = _mm256_add_ps(_mm512_castps512_ps256(zmm[15]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[18], 1);
            ymm[14] = _mm256_add_ps(_mm512_castps512_ps256(zmm[18]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            ymm[0] = _mm256_hadd_ps(ymm[5], ymm[8]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[11], ymm[14]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[5] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);
            xmm[5] = _mm_mul_ps(xmm[5], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);
            xmm[1] = _mm_loadu_ps(c_temp + 1*rs_c0);
            xmm[5] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[5]);
            _mm_storeu_ps(c_temp + 1*rs_c0, xmm[5]);

            c_temp += 2*rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}

LPGEMM_M_RD_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x32_rd)
{
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

    __m512 zmm[32];
    __m256 ymm[16];
    __m128 xmm[16];

    dim_t jj;
    for ( jj = 0; jj < 32; jj += 4 )    // SLOOP3X4J
    {
        float* c_temp = cbuf;
        float* a_temp = abuf;
        float* b_temp = bbuf;

        c_temp += jj * cs_c0;
        b_temp += jj * cs_b0;

        // zero out all zmm registers
        for ( int i = 0; i < 32; ++i )
        {
            zmm[i] = _mm512_setzero_ps();
        }
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

        for ( dim_t k_iterator = 0; k_iterator < k_iter64; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 4; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter32; ++k_iterator )
        {
            for ( dim_t unroll = 0; unroll < 2; ++unroll )
            {
                zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

                zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
                zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
                zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
                zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

                zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
                zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

                a_temp += 16;
                b_temp += 16;
            }
        }

        for ( dim_t k_iterator = 0; k_iterator < k_iter16; ++k_iterator )
        {
            zmm[0]  = _mm512_loadu_ps(a_temp + 0*rs_a0);

            zmm[6]  = _mm512_loadu_ps(b_temp + 0*cs_b0);
            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 1*cs_b0);
            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 2*cs_b0);
            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

            zmm[6]  = _mm512_loadu_ps(b_temp + 3*cs_b0);
            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);

            a_temp += 16;
            b_temp += 16;
        }

        if ( k_left16 != 0 )
        {
            __mmask16 m_mask = (1 << (k_left16)) - 1;

            zmm[0]  = _mm512_maskz_loadu_ps(m_mask, a_temp + 0*rs_a0);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 0*cs_b0);

            zmm[8]  = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[8]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 1*cs_b0);

            zmm[11] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[11]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 2*cs_b0);

            zmm[14] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[14]);

            zmm[6]  = _mm512_maskz_loadu_ps(m_mask, b_temp + 3*cs_b0);

            zmm[17] = _mm512_fmadd_ps(zmm[0], zmm[6], zmm[17]);
        }

        if ( beta == 0 )
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

            c_temp += rs_c0;
        }
        else
        {
            ymm[0] = _mm512_extractf32x8_ps(zmm[8], 1);
            ymm[4] = _mm256_add_ps(_mm512_castps512_ps256(zmm[8]), ymm[0]);
            ymm[1] = _mm512_extractf32x8_ps(zmm[11], 1);
            ymm[7] = _mm256_add_ps(_mm512_castps512_ps256(zmm[11]), ymm[1]);
            ymm[2] = _mm512_extractf32x8_ps(zmm[14], 1);
            ymm[10] = _mm256_add_ps(_mm512_castps512_ps256(zmm[14]), ymm[2]);
            ymm[3] = _mm512_extractf32x8_ps(zmm[17], 1);
            ymm[13] = _mm256_add_ps(_mm512_castps512_ps256(zmm[17]), ymm[3]);

            ymm[0] = _mm256_hadd_ps(ymm[4], ymm[7]);
            xmm[1] = _mm256_extractf128_ps(ymm[0], 1);
            xmm[0] = _mm_add_ps(_mm256_castps256_ps128(ymm[0]), xmm[1]);
            ymm[2] = _mm256_hadd_ps(ymm[10], ymm[13]);
            xmm[3] = _mm256_extractf128_ps(ymm[2], 1);
            xmm[2] = _mm_add_ps(_mm256_castps256_ps128(ymm[2]), xmm[3]);
            xmm[4] = _mm_hadd_ps(xmm[0], xmm[2]);

            // ALPHA SCAL
            xmm[0] = _mm_broadcast_ss(&alpha);
            xmm[4] = _mm_mul_ps(xmm[4], xmm[0]);

            // C STORE
            xmm[0] = _mm_broadcast_ss(&beta);
            xmm[1] = _mm_loadu_ps(c_temp + 0*rs_c0);
            xmm[4] = _mm_fmadd_ps(xmm[0], xmm[1], xmm[4]);
            _mm_storeu_ps(c_temp + 0*rs_c0, xmm[4]);

            c_temp += rs_c0;
        }

        c_temp += 6*rs_c0;
        a_temp += 6*rs_a0;
    }
}
#endif
