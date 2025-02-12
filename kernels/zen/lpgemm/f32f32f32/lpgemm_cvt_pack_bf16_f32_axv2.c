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

#include <immintrin.h>
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_kernel_macros_f32_avx2.h"

#define CVT_BF16_F32_SHIFT_AVX2_lt8( reg, k_left, ic, kr) { \
        int16_t data_feeder[8] = {0};  \
        for( dim_t i = 0; i < k_left; i++) \
        { \
            data_feeder[i] = *(a + (ic * rs_a) + (kr * cs_a) + i);  \
        } \
        reg = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                                    (const __m128i*)data_feeder) ); \
}

#define LOAD_AND_CONVERT_BF16_F32(reg, ic )   \
        reg = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)( a + ( ic * rs_a ) + ( kr * cs_a ) ) ) );

void cvt_bf16_f32_row_major
    (
      float*	      cvt_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      const dim_t     rs_p,
      const dim_t     cs_p
    )
{
    dim_t MR = 16;
    dim_t k_left = KC % 8;

    __m256i store_mask;

    __m256 a_reg[16];

    dim_t ic = 0, kr = 0;

    for( ic = 0; ( ic + MR - 1 ) < MC; ic += MR )
	{
		for( kr = 0; ( kr + 8 - 1) < KC; kr += 8 )
		{
            /*Load 8 BF16 elements from 16 rows, and convert them to F32 elements*/
            LOAD_AND_CONVERT_BF16_F32(a_reg[0], ( ic + 0 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[1], ( ic + 1 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[2], ( ic + 2 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[3], ( ic + 3 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[4], ( ic + 4 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[5], ( ic + 5 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[6], ( ic + 6 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[7], ( ic + 7 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[8], ( ic + 8 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[9], ( ic + 9 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[10], ( ic + 10 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[11], ( ic + 11 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[12], ( ic + 12 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[13], ( ic + 13 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[14], ( ic + 14 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[15], ( ic + 15 ) );

            /*Store 8 F32 elements each in 16 rows */
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , a_reg[0] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr , a_reg[1] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr , a_reg[2] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr , a_reg[3] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 4 ) * rs_p ) + kr , a_reg[4] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 5 ) * rs_p ) + kr , a_reg[5] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 6 ) * rs_p ) + kr , a_reg[6] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 7 ) * rs_p ) + kr , a_reg[7] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 8 ) * rs_p ) + kr , a_reg[8] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 9 ) * rs_p ) + kr , a_reg[9] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 10 ) * rs_p ) + kr , a_reg[10] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 11 ) * rs_p ) + kr , a_reg[11] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 12 ) * rs_p ) + kr , a_reg[12] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 13 ) * rs_p ) + kr , a_reg[13] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 14 ) * rs_p ) + kr , a_reg[14] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 15 ) * rs_p ) + kr , a_reg[15] );
        }
        if( k_left > 0)
        {
            /*Using a data_feeder function to load < 8 elemnts and convert
            to f32 elements*/
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],k_left,( ic + 0 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[1],k_left,( ic + 1 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[2],k_left,( ic + 2 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[3],k_left,( ic + 3 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[4],k_left,( ic + 4 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[5],k_left,( ic + 5 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[6],k_left,( ic + 6 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[7],k_left,( ic + 7 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[8],k_left,( ic + 8 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[9],k_left,( ic + 9 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[10],k_left,( ic + 10 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[11],k_left,( ic + 11 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[12],k_left,( ic + 12 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[13],k_left,( ic + 13 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[14],k_left,( ic + 14 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[15],k_left,( ic + 15 ), kr);

            GET_STORE_MASK(k_left, store_mask);

            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr, store_mask, a_reg[1] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr, store_mask, a_reg[2] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr, store_mask, a_reg[3] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 4 ) * rs_p ) + kr, store_mask, a_reg[4] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 5 ) * rs_p ) + kr, store_mask, a_reg[5] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 6 ) * rs_p ) + kr, store_mask, a_reg[6] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 7 ) * rs_p ) + kr, store_mask, a_reg[7] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 8 ) * rs_p ) + kr, store_mask, a_reg[8] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 9 ) * rs_p ) + kr, store_mask, a_reg[9] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 10 ) * rs_p ) + kr, store_mask, a_reg[10] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 11 ) * rs_p ) + kr, store_mask, a_reg[11] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 12 ) * rs_p ) + kr, store_mask, a_reg[12] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 13 ) * rs_p ) + kr, store_mask, a_reg[13] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 14 ) * rs_p ) + kr, store_mask, a_reg[14] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 15 ) * rs_p ) + kr, store_mask, a_reg[15] );
        }
    }
    for( ; ( ic + 8 - 1 ) < MC; ic += 8 )
    {
        for( kr = 0; ( kr + 8 - 1 ) < KC; kr += 8 )
        {
            LOAD_AND_CONVERT_BF16_F32(a_reg[0], ( ic + 0 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[1], ( ic + 1 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[2], ( ic + 2 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[3], ( ic + 3 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[4], ( ic + 4 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[5], ( ic + 5 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[6], ( ic + 6 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[7], ( ic + 7 ) );

            _mm256_storeu_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , a_reg[0] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr , a_reg[1] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr , a_reg[2] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr , a_reg[3] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 4 ) * rs_p ) + kr , a_reg[4] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 5 ) * rs_p ) + kr , a_reg[5] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 6 ) * rs_p ) + kr , a_reg[6] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 7 ) * rs_p ) + kr , a_reg[7] );
        }
        if(k_left)
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],k_left,( ic + 0 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[1],k_left,( ic + 1 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[2],k_left,( ic + 2 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[3],k_left,( ic + 3 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[4],k_left,( ic + 4 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[5],k_left,( ic + 5 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[6],k_left,( ic + 6 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[7],k_left,( ic + 7 ), kr);

            GET_STORE_MASK(k_left, store_mask);

            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr, store_mask, a_reg[1] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr, store_mask , a_reg[2] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr, store_mask , a_reg[3] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 4 ) * rs_p ) + kr, store_mask , a_reg[4] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 5 ) * rs_p ) + kr, store_mask , a_reg[5] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 6 ) * rs_p ) + kr, store_mask , a_reg[6] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 7 ) * rs_p ) + kr, store_mask , a_reg[7] );
        }
    }
    for( ; ( ic + 4 - 1 ) < MC; ic += 4 )
    {
        for( kr = 0; ( kr + 8 - 1 ) < KC; kr += 8 )
        {
            LOAD_AND_CONVERT_BF16_F32(a_reg[0], ( ic + 0 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[1], ( ic + 1 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[2], ( ic + 2 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[3], ( ic + 3 ) );

            _mm256_storeu_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , a_reg[0] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr , a_reg[1] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr , a_reg[2] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr , a_reg[3] );
        }

        if( k_left > 0 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],k_left,( ic + 0 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[1],k_left,( ic + 1 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[2],k_left,( ic + 2 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[3],k_left,( ic + 3 ), kr);

            GET_STORE_MASK(k_left, store_mask);

            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr, store_mask, a_reg[1] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr, store_mask , a_reg[2] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr, store_mask , a_reg[3] );
        }
    }
    for( ; ( ic + 2 - 1 ) < MC; ic += 2 )
    {
        for( kr = 0; ( kr + 8 - 1 ) < KC; kr += 8 )
        {
            LOAD_AND_CONVERT_BF16_F32(a_reg[0], ( ic + 0 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[1], ( ic + 1 ) );

            _mm256_storeu_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , a_reg[0] );
            _mm256_storeu_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr , a_reg[1] );
        }

        if( k_left > 0 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],k_left,( ic + 0 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[1],k_left,( ic + 1 ), kr);

            GET_STORE_MASK(k_left, store_mask);

            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr, store_mask, a_reg[1] );
        }
    }
    for( ; ( ic ) < MC; ic += 1 )
    {
        for( kr = 0; ( kr + 8 - 1 ) < KC; kr += 8 )
        {
            LOAD_AND_CONVERT_BF16_F32(a_reg[0], ( ic + 0 ) );

            _mm256_storeu_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , a_reg[0] );
        }
        for( ; ( kr + 4 - 1 ) < KC; kr += 4 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],4,( ic + 0 ), kr);

            GET_STORE_MASK(4, store_mask);

            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , store_mask, a_reg[0] );
        }
        for( ; ( kr + 2 - 1 ) < KC; kr += 2 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],2,( ic + 0 ), kr);

            GET_STORE_MASK(2, store_mask);

            _mm256_maskstore_ps ( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , store_mask, a_reg[0] );
        }
        for( ; ( kr ) < KC; kr += 1 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],1,( ic + 0 ), kr);

            GET_STORE_MASK(2, store_mask);

            _mm256_maskstore_ps( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr , store_mask, a_reg[0] );
        }
    }
}

#define LOAD_1BF16_ELEMENT(a_ptr,kr,reg)   \
{    \
    bfloat16 buff[8] = {0};  \
    for( dim_t i = 0; i < 1; i++ ) buff[i] = *(a_ptr + ( kr * cs_a ) + i);  \
    reg = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)buff) );  \
}

#define LOAD_AND_CONVERT_8COLS_BF16_F32(kr)    \
{   \
    bfloat16 *a_ptr = (bfloat16*)( a + ( ic * rs_a ) );   \
    a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 0 ) * cs_a ) ) ) ); \
    a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 1 ) * cs_a ) ) ) );  \
    a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 2 ) * cs_a ) ) ) );  \
    a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 3 ) * cs_a ) ) ) );  \
    a_reg[4] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 4 ) * cs_a ) ) ) ); \
    a_reg[5] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 5 ) * cs_a ) ) ) );  \
    a_reg[6] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 6 ) * cs_a ) ) ) );  \
    a_reg[7] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 7 ) * cs_a ) ) ) );  \
}

#define MASKED_LOAD_AND_CONVERT_8COLS_BF16_F32(kr, mask)    \
{   \
    bfloat16 *a_ptr = (bfloat16*)( a + ( ic * rs_a ) );   \
    a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 0 ) * cs_a ) ), mask ) ); \
    a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 1 ) * cs_a ) ), mask ) );  \
    a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 2 ) * cs_a ) ), mask ) );  \
    a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 3 ) * cs_a ) ), mask ) );  \
    a_reg[4] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 4 ) * cs_a ) ), mask ) ); \
    a_reg[5] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 5 ) * cs_a ) ), mask ) );  \
    a_reg[6] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 6 ) * cs_a ) ), mask ) );  \
    a_reg[7] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
        (int const*)(a_ptr + ( ( kr + 7 ) * cs_a ) ), mask ) );  \
}

#define LOAD_AND_CONVERT_8COLS_1ELE_BF16_F32(kr)    \
{   \
    bfloat16 *a_ptr = (bfloat16*)( a + ( ic * rs_a ) );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 0 ), a_reg[0] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 1 ), a_reg[1] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 2 ), a_reg[2] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 3 ), a_reg[3] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 4 ), a_reg[4] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 5 ), a_reg[5] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 6 ), a_reg[6] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 7 ), a_reg[7] );   \
}

#define LOAD_AND_CONVERT_4COLS_BF16_F32(kr)    \
{   \
    bfloat16 *a_ptr = (bfloat16*)( a + ( ic * rs_a ) );   \
    a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 0 ) * cs_a ) ) ) ); \
    a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 1 ) * cs_a ) ) ) );  \
    a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 2 ) * cs_a ) ) ) );  \
    a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(a_ptr + ( ( kr + 3 ) * cs_a ) ) ) );  \
}

#define MASKED_LOAD_AND_CONVERT_4COLS_BF16_F32(kr, mask)    \
{   \
    bfloat16 *a_ptr = (bfloat16*)( a + ( ic * rs_a ) );   \
    a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
                (int const*)(a_ptr + ( ( kr + 0 ) * cs_a ) ), mask ) ); \
    a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
                (int const*)(a_ptr + ( ( kr + 1 ) * cs_a ) ), mask ) );  \
    a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
                (int const*)(a_ptr + ( ( kr + 2 ) * cs_a ) ), mask ) );  \
    a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
                (int const*)(a_ptr + ( ( kr + 3 ) * cs_a ) ), mask ) );  \
}

#define LOAD_AND_CONVERT_4COLS_1ELE_BF16_F32(kr)    \
{   \
    bfloat16 *a_ptr = (bfloat16*)( a + ( ic * rs_a ) );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 0 ), a_reg[0] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 1 ), a_reg[1] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 2 ), a_reg[2] );   \
    LOAD_1BF16_ELEMENT(a_ptr, ( kr + 3 ), a_reg[3] );   \
}

#define MASKED_LOAD_AND_CONVERT_2COLS_BF16_F32(kr, mask)    \
{   \
    bfloat16 *a_ptr = (bfloat16*)( a + ( ic * rs_a ) );   \
    a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
                (int const*)(a_ptr + ( ( kr + 0 ) * cs_a ) ), mask ) ); \
    a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32( \
                (int const*)(a_ptr + ( ( kr + 1 ) * cs_a ) ), mask ) );  \
}

#define UNPACKLO8x8_AVX2  \
	b_reg[0] =_mm256_unpacklo_ps( a_reg[0], a_reg[1] );  \
	b_reg[1] =_mm256_unpacklo_ps( a_reg[2], a_reg[3] ); \
	b_reg[2] =_mm256_unpacklo_ps( a_reg[4], a_reg[5] );  \
	b_reg[3] =_mm256_unpacklo_ps( a_reg[6], a_reg[7] ); \

#define UNPACKHI8x8_AVX2  \
	b_reg[4] = _mm256_unpackhi_ps( a_reg[0], a_reg[1] );  \
	b_reg[5] = _mm256_unpackhi_ps( a_reg[2], a_reg[3] ); \
	b_reg[6] = _mm256_unpackhi_ps( a_reg[4], a_reg[5] );  \
	b_reg[7] = _mm256_unpackhi_ps( a_reg[6], a_reg[7] );

#define SHUFFLE_8x8_AVX2  \
    a_reg[0] = _mm256_shuffle_ps( b_reg[0], b_reg[1], 0x44 );  \
    a_reg[1] = _mm256_shuffle_ps( b_reg[0], b_reg[1], 0xEE );  \
    a_reg[2] = _mm256_shuffle_ps( b_reg[2], b_reg[3], 0x44 );  \
    a_reg[3] = _mm256_shuffle_ps( b_reg[2], b_reg[3], 0xEE );  \
\
    a_reg[4] = _mm256_shuffle_ps( b_reg[4], b_reg[5], 0x44 );  \
    a_reg[5] = _mm256_shuffle_ps( b_reg[4], b_reg[5], 0xEE );  \
    a_reg[6] = _mm256_shuffle_ps( b_reg[6], b_reg[7], 0x44 );  \
    a_reg[7] = _mm256_shuffle_ps( b_reg[6], b_reg[7], 0xEE );  \

#define PERMUTE_8x8_AVX2  \
    b_reg[0] = _mm256_permute2f128_ps( a_reg[0], a_reg[2], 0x20 );  \
    b_reg[1] = _mm256_permute2f128_ps( a_reg[1], a_reg[3], 0x20 );  \
    b_reg[2] = _mm256_permute2f128_ps( a_reg[4], a_reg[6], 0x20 );  \
    b_reg[3] = _mm256_permute2f128_ps( a_reg[5], a_reg[7], 0x20 );  \
\
    b_reg[4] = _mm256_permute2f128_ps( a_reg[0], a_reg[2], 0x31 );  \
    b_reg[5] = _mm256_permute2f128_ps( a_reg[1], a_reg[3], 0x31 );  \
    b_reg[6] = _mm256_permute2f128_ps( a_reg[4], a_reg[6], 0x31 );  \
    b_reg[7] = _mm256_permute2f128_ps( a_reg[5], a_reg[7], 0x31 );  \

#define STORE_8COLS_AVX2 \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr ), b_reg[0] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr ), b_reg[1] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr ), b_reg[2] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr ), b_reg[3] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 4 ) * rs_p ) + kr ), b_reg[4] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 5 ) * rs_p ) + kr ), b_reg[5] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 6 ) * rs_p ) + kr ), b_reg[6] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 7 ) * rs_p ) + kr ), b_reg[7] );    \

#define STORE_4COLS_AVX2 \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr ), b_reg[0] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr ), b_reg[1] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr ), b_reg[2] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr ), b_reg[3] );    \

#define STORE_2COLS_AVX2 \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr ), b_reg[0] );    \
    _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr ), b_reg[1] );

    #define MASKED_STORE_F32COLS_AVX2(mask)  \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr ), mask, b_reg[0] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr ), mask, b_reg[1] );  \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr ), mask, b_reg[2] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr ), mask, b_reg[3] );  \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 4 ) * rs_p ) + kr ), mask, b_reg[4] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 5 ) * rs_p ) + kr ), mask, b_reg[5] );  \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 6 ) * rs_p ) + kr ), mask, b_reg[6] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 7 ) * rs_p ) + kr ), mask, b_reg[7] );

#define MASKED_STORE_4COLS_AVX2(mask) \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr ), mask, b_reg[0] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr ), mask, b_reg[1] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 2 ) * rs_p ) + kr ), mask, b_reg[2] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 3 ) * rs_p ) + kr ), mask, b_reg[3] );    \

 #define MASKED_STORE_2COLS_AVX2(mask) \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 0 ) * rs_p ) + kr ), mask, b_reg[0] );    \
    _mm256_maskstore_ps( ( cvt_buffer + ( ( ic + 1 ) * rs_p ) + kr ), mask, b_reg[1] );    \

void cvt_bf16_f32_col_major
(
    float*	      cvt_buffer,
    const bfloat16* a,
    const dim_t     rs_a,
    const dim_t     cs_a,
    const dim_t     MC,
    const dim_t     KC,
    const dim_t     rs_p,
    const dim_t     cs_p
)
{
    dim_t MR = 8;
    dim_t ic, kr;

    __m256 a_reg[8], b_reg[8];
    a_reg[0] = _mm256_setzero_ps();
    a_reg[1] = _mm256_setzero_ps();
    a_reg[2] = _mm256_setzero_ps();
    a_reg[3] = _mm256_setzero_ps();
    a_reg[4] = _mm256_setzero_ps();
    a_reg[5] = _mm256_setzero_ps();
    a_reg[6] = _mm256_setzero_ps();
    a_reg[7] = _mm256_setzero_ps();

    for( ic = 0; ( ic + MR - 1 ) < MC; ic += MR )
    {
        for( kr = 0; ( kr + 7 ) < KC; kr += 8)
        {
            LOAD_AND_CONVERT_8COLS_BF16_F32(kr)
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2

            STORE_8COLS_AVX2;
        }
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            __m256i store_mask;

            LOAD_AND_CONVERT_4COLS_BF16_F32(kr)
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2

            GET_STORE_MASK(4, store_mask);
            MASKED_STORE_F32COLS_AVX2(store_mask)

        }
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            __m256i store_mask;
            a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(( a + ( ic * rs_a ) ) + ( ( kr + 0 ) * cs_a ) ) ) ); \
            a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(( a + ( ic * rs_a ) ) + ( ( kr + 1 ) * cs_a ) ) ) );  \
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2

            GET_STORE_MASK(2, store_mask);
            MASKED_STORE_F32COLS_AVX2(store_mask)
        }
        for( ; kr < KC; kr += 1 )
        {
            __m256i store_mask;
            a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
                (const __m128i*)(( a + ( ic * rs_a ) ) + ( ( kr + 0 ) * cs_a ) ) ) ); \

            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2

            GET_STORE_MASK(1, store_mask);
            MASKED_STORE_F32COLS_AVX2(store_mask)
        }
    }
    for( ; ( ic + 3 ) < MC; ic += 4 )
    {
        __m128i load_mask = _mm_set_epi16(0, 0, 0, 0, -1, -1, -1, -1);
        for( kr = 0; ( kr + 7 ) < KC; kr += 8 )
        {
            MASKED_LOAD_AND_CONVERT_8COLS_BF16_F32( kr, load_mask );

            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            STORE_4COLS_AVX2
        }
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            __m256i store_mask;

            MASKED_LOAD_AND_CONVERT_4COLS_BF16_F32( kr, load_mask );

            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(4, store_mask)
            MASKED_STORE_4COLS_AVX2(store_mask);
        }
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            __m256i store_mask;

            MASKED_LOAD_AND_CONVERT_2COLS_BF16_F32(kr, load_mask);
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(2, store_mask)
            MASKED_STORE_4COLS_AVX2(store_mask);
        }
        for( ; kr < KC ; kr += 1 )
        {
            __m256i store_mask;

            a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32(
                (int const*)(( a + ( ic * rs_a ) ) + ( ( kr + 0 ) * cs_a ) ), load_mask ) );
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(1, store_mask)
            MASKED_STORE_4COLS_AVX2(store_mask);
        }
    }
    for( ; ( ic + 1 ) < MC; ic += 2 )
    {
        __m128i load_mask = _mm_set_epi16(0, 0, 0, 0, -0, 0, -1, -1);
        for( kr = 0; ( kr + 7 ) < KC; kr += 8 )
        {
            MASKED_LOAD_AND_CONVERT_8COLS_BF16_F32( kr, load_mask );
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            STORE_2COLS_AVX2
        }
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            __m256i store_mask;
            MASKED_LOAD_AND_CONVERT_4COLS_BF16_F32(kr, load_mask);
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(4, store_mask);
            MASKED_STORE_2COLS_AVX2(store_mask);
        }
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            __m256i store_mask;
            MASKED_LOAD_AND_CONVERT_2COLS_BF16_F32(kr, load_mask);
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(2, store_mask);
            MASKED_STORE_2COLS_AVX2(store_mask);
        }
        for( ; kr < KC; kr += 1 )
        {
            __m256i store_mask;
            a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_maskload_epi32(
                (int const*)(( a + ( ic * rs_a ) ) + ( ( kr + 0 ) * cs_a ) ), load_mask ) );
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(1, store_mask);
            MASKED_STORE_2COLS_AVX2(store_mask);
        }
    }
    for( ; ( ic ) < MC; ic += 1 )
    {
        for( kr = 0; ( kr + 7 ) < KC; kr += 8 )
        {
            LOAD_AND_CONVERT_8COLS_1ELE_BF16_F32(kr)
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            _mm256_storeu_ps( ( cvt_buffer + ( ( ic + 0 ) * KC ) + kr ), b_reg[0] );
        }
        for( ; ( kr + 3 ) < KC; kr += 4 )
        {
            __m256i store_mask;
            LOAD_AND_CONVERT_4COLS_1ELE_BF16_F32(kr)
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(4, store_mask);
            MASKED_STORE_2COLS_AVX2(store_mask);
        }
        for( ; ( kr + 1 ) < KC; kr += 2 )
        {
            __m256i store_mask;

            LOAD_1BF16_ELEMENT( (bfloat16*)( a + ( ic * rs_a ) ), ( kr + 0 ), a_reg[0] );
            LOAD_1BF16_ELEMENT( (bfloat16*)( a + ( ic * rs_a ) ), ( kr + 1 ), a_reg[1] );

            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(2, store_mask);
            MASKED_STORE_2COLS_AVX2(store_mask);
        }
        for( ; kr < KC; kr += 1 )
        {
            __m256i store_mask;

            LOAD_1BF16_ELEMENT( (bfloat16*)( a + ( ic * rs_a ) ), ( kr + 0 ), a_reg[0] );
            UNPACKLO8x8_AVX2
            UNPACKHI8x8_AVX2
            SHUFFLE_8x8_AVX2
            PERMUTE_8x8_AVX2
            GET_STORE_MASK(1, store_mask);
            MASKED_STORE_2COLS_AVX2(store_mask);
        }
    }
}
void cvt_bf16_f32(
      float*	      cvt_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      const dim_t     rs_p,
      const dim_t     cs_p
    )
{
    if( cs_a == 1 )
    {
        cvt_bf16_f32_row_major( cvt_buffer, a, rs_a, cs_a, MC, KC, rs_p, cs_p );
    }
    else
    {
        cvt_bf16_f32_col_major( cvt_buffer, a, rs_a, cs_a, MC, KC, rs_p, cs_p );
    }
}
#endif