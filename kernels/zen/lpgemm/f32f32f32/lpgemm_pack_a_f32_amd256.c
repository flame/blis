/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include <immintrin.h>
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#define UNPACKLO_PS8 \
    b_reg[0] = _mm256_unpacklo_ps( a_reg[0], a_reg[1] ); \
    b_reg[1] = _mm256_unpacklo_ps( a_reg[2], a_reg[3] ); \
    b_reg[2] = _mm256_unpacklo_ps( a_reg[4], a_reg[5] ); \
    b_reg[3] = _mm256_unpacklo_ps( a_reg[6], a_reg[7] );

#define UNPACKHI_PS8 \
    b_reg[4] = _mm256_unpackhi_ps( a_reg[0], a_reg[1] ); \
    b_reg[5] = _mm256_unpackhi_ps( a_reg[2], a_reg[3] ); \
    b_reg[6] = _mm256_unpackhi_ps( a_reg[4], a_reg[5] ); \
    b_reg[7] = _mm256_unpackhi_ps( a_reg[6], a_reg[7] );

#define UNPACKLO_PD8 \
    a_reg[0] = (__m256)_mm256_unpacklo_pd( (__m256d)b_reg[0], (__m256d)b_reg[1] ); \
    a_reg[1] = (__m256)_mm256_unpacklo_pd( (__m256d)b_reg[2], (__m256d)b_reg[3] ); \
    a_reg[2] = (__m256)_mm256_unpacklo_pd( (__m256d)b_reg[4], (__m256d)b_reg[5] ); \
    a_reg[3] = (__m256)_mm256_unpacklo_pd( (__m256d)b_reg[6], (__m256d)b_reg[7] );

#define UNPACKHI_PD8 \
    a_reg[4] = (__m256)_mm256_unpackhi_pd( (__m256d)b_reg[0], (__m256d)b_reg[1] ); \
    a_reg[5] = (__m256)_mm256_unpackhi_pd( (__m256d)b_reg[2], (__m256d)b_reg[3] ); \
    a_reg[6] = (__m256)_mm256_unpackhi_pd( (__m256d)b_reg[4], (__m256d)b_reg[5] ); \
    a_reg[7] = (__m256)_mm256_unpackhi_pd( (__m256d)b_reg[6], (__m256d)b_reg[7] );

#define PERMUTE2F128_PS8 \
    b_reg[0] = _mm256_permute2f128_ps( a_reg[0], a_reg[1], 0x20 ); \
    b_reg[1] = _mm256_permute2f128_ps( a_reg[4], a_reg[5], 0x20 ); \
    b_reg[2] = _mm256_permute2f128_ps( a_reg[2], a_reg[3], 0x20 ); \
    b_reg[3] = _mm256_permute2f128_ps( a_reg[6], a_reg[7], 0x20 ); \
    b_reg[4] = _mm256_permute2f128_ps( a_reg[0], a_reg[1], 0x31 ); \
    b_reg[5] = _mm256_permute2f128_ps( a_reg[4], a_reg[5], 0x31 ); \
    b_reg[6] = _mm256_permute2f128_ps( a_reg[2], a_reg[3], 0x31 ); \
    b_reg[7] = _mm256_permute2f128_ps( a_reg[6], a_reg[7], 0x31 );

void packa_mr8_f32f32f32of32_col_major
(
  float*	      pack_a_buffer,
  const float* a,
  const dim_t     rs_a,
  const dim_t     cs_a,
  const dim_t     MC,
  const dim_t     KC,
  dim_t*          rs_p,
  dim_t*          cs_p
)
{
    dim_t MR = 8;
    dim_t ic, kr;

    __m256 a_reg[8], b_reg[8];

    __m256i k_masks[3] = {
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
        _mm256_set_epi32( 0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
    };

    __m256i load_mask, store_mask;

    // These registers are set with zeroes to avoid compiler warnings
    // To-DO: TO be removed when pack code is optimized for fringe cases.
    a_reg[0] = _mm256_setzero_ps();
    a_reg[1] = _mm256_setzero_ps();
    a_reg[2] = _mm256_setzero_ps();
    a_reg[3] = _mm256_setzero_ps();
    a_reg[4] = _mm256_setzero_ps();
    a_reg[5] = _mm256_setzero_ps();
    a_reg[6] = _mm256_setzero_ps();
    a_reg[7] = _mm256_setzero_ps();

    for( ic = 0; ( ic + MR -1 ) < MC; ic += MR )
    {
        for( kr = 0; ( kr + 7 ) < KC; kr += 8 )
        {
            // Transposing the 8x8 block of data
            a_reg[0] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
            a_reg[1] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
            a_reg[2] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
            a_reg[3] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );
            a_reg[4] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ) );
            a_reg[5] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ) );
            a_reg[6] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ) );
            a_reg[7] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ) );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), b_reg[0] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), b_reg[1] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), b_reg[2] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), b_reg[3] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), b_reg[4] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), b_reg[5] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), b_reg[6] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), b_reg[7] );
        }
        store_mask = k_masks[2]; // mask to store 4 elements
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            // Transposing the 8x8 block of data
            a_reg[0] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
            a_reg[1] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
            a_reg[2] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
            a_reg[3] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), store_mask, b_reg[2] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), store_mask, b_reg[3] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), store_mask, b_reg[4] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), store_mask, b_reg[5] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), store_mask, b_reg[6] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), store_mask, b_reg[7] );
        }
        store_mask = k_masks[1]; // mask to store 2 elements
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            // Transposing the 8x8 block of data
            a_reg[0] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
            a_reg[1] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), store_mask, b_reg[2] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), store_mask, b_reg[3] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), store_mask, b_reg[4] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), store_mask, b_reg[5] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), store_mask, b_reg[6] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), store_mask, b_reg[7] );
        }
        store_mask = k_masks[0]; // mask to store 1 element
        for( ; ( kr + 0 ) < KC; kr += 1 )
        {
            // Transposing the 8x8 block of data
            a_reg[0] = _mm256_loadu_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), store_mask, b_reg[2] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), store_mask, b_reg[3] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), store_mask, b_reg[4] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), store_mask, b_reg[5] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), store_mask, b_reg[6] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), store_mask, b_reg[7] );
        }
    }
    for( ; ( ic + 3 ) < MC; ic += 4 )
    {
        load_mask = k_masks[2]; // mask to load 4 elements
        for( kr = 0; ( kr + 7 ) < KC; kr += 8 )
        {
            // Transposing the 8x8 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );
            a_reg[2] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ), load_mask );
            a_reg[3] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ), load_mask );
            a_reg[4] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ), load_mask );
            a_reg[5] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ), load_mask );
            a_reg[6] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ), load_mask );
            a_reg[7] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), b_reg[0] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), b_reg[1] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), b_reg[2] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), b_reg[3] );
        }
        store_mask = k_masks[2]; // mask to store 4 elements
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            // Transposing the 4x4 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );
            a_reg[2] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ), load_mask );
            a_reg[3] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), store_mask, b_reg[2] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), store_mask, b_reg[3] );
        }
        store_mask = k_masks[1]; // mask to store 2 elements
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            // transposing the 4x2 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8
            UNPACKLO_PD8
            UNPACKHI_PD8
            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), store_mask, b_reg[2] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), store_mask, b_reg[3] );
        }
        store_mask = k_masks[0]; // mask to store 1 element
        for( ; ( kr + 0 ) < KC; kr += 1 )
        {
            // transposing the 4x1 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8
            UNPACKLO_PD8
            UNPACKHI_PD8
            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), store_mask, b_reg[2] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), store_mask, b_reg[3] );
        }
    }
    for( ; ( ic + 1 ) < MC; ic += 2 )
    {
        load_mask = k_masks[1]; // mask to load 2 elements
        for( kr = 0; ( kr + 7 ) < KC; kr += 8 )
        {
            // Transposing the 2x8 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );
            a_reg[2] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ), load_mask );
            a_reg[3] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ), load_mask );
            a_reg[4] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ), load_mask );
            a_reg[5] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ), load_mask );
            a_reg[6] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ), load_mask );
            a_reg[7] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), b_reg[0] );
            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), b_reg[1] );
        }
        store_mask = k_masks[2]; // mask to store 4 elements
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            // Transposing the 2x4 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );
            a_reg[2] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ), load_mask );
            a_reg[3] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
        }
        store_mask = k_masks[1]; // mask to store 2 elements
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            // Transposing the 2x2 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
        }
        store_mask = k_masks[0]; // mask to store 1 element
        for( ; ( kr + 0 ) < KC; kr += 1 )
        {
            // Transposing the 2x1 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), store_mask, b_reg[1] );
        }
    }
    for( ; ( ic + 0 ) < MC; ic += 1 )
    {
        load_mask = k_masks[0]; // mask to load 1 element
        for( kr = 0; ( kr + 7 ) < KC; kr += 8 )
        {
            // Transposing the 1x8 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );
            a_reg[2] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ), load_mask );
            a_reg[3] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ), load_mask );
            a_reg[4] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ), load_mask );
            a_reg[5] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ), load_mask );
            a_reg[6] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ), load_mask );
            a_reg[7] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_storeu_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), b_reg[0] );
        }
        store_mask = k_masks[2]; // mask to store 4 elements
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            // Transposing the 1x4 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );
            a_reg[2] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ), load_mask );
            a_reg[3] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
        }
        store_mask = k_masks[1]; // mask to store 2 elements
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            // Transposing the 1x2 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );
            a_reg[1] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
        }
        store_mask = k_masks[0]; // mask to store 1 element
        for( ; ( kr + 0 ) < KC; kr += 1 )
        {
            // Transposing the 1x1 block of data
            a_reg[0] = _mm256_maskload_ps( ( float const* )( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ), load_mask );

            UNPACKLO_PS8
            UNPACKHI_PS8

            UNPACKLO_PD8
            UNPACKHI_PD8

            PERMUTE2F128_PS8

            _mm256_maskstore_ps( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), store_mask, b_reg[0] );
        }
    }
    // Set the row and column strides of the packed matrix.
    *rs_p = KC;
    *cs_p = 1;
}

#endif // BLIS_ADDON_LPGEMM
