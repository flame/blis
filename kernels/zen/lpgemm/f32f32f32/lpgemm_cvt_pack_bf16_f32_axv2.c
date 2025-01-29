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

static inline void get_store_mask( __m256i *mask, dim_t k )
{
    int32_t mask_vec[8] = {0};
    for( dim_t i = 0; i < k; i++ ) mask_vec[i] = -1;
    *mask = _mm256_loadu_si256((__m256i const *)mask_vec);
}

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

void cvt_pack_bf16_f32_row_major
    (
      float*	      pack_a_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
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
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr , a_reg[2] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr , a_reg[3] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr , a_reg[4] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr , a_reg[5] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr , a_reg[6] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr , a_reg[7] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 8 ) * KC ) + kr , a_reg[8] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 9 ) * KC ) + kr , a_reg[9] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 10 ) * KC ) + kr , a_reg[10] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 11 ) * KC ) + kr , a_reg[11] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 12 ) * KC ) + kr , a_reg[12] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 13 ) * KC ) + kr , a_reg[13] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 14 ) * KC ) + kr , a_reg[14] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 15 ) * KC ) + kr , a_reg[15] );
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

            get_store_mask(&store_mask, k_left);

            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, store_mask, a_reg[1] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr, store_mask, a_reg[2] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr, store_mask, a_reg[3] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr, store_mask, a_reg[4] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr, store_mask, a_reg[5] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr, store_mask, a_reg[6] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr, store_mask, a_reg[7] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 8 ) * KC ) + kr, store_mask, a_reg[8] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 9 ) * KC ) + kr, store_mask, a_reg[9] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 10 ) * KC ) + kr, store_mask, a_reg[10] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 11 ) * KC ) + kr, store_mask, a_reg[11] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 12 ) * KC ) + kr, store_mask, a_reg[12] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 13 ) * KC ) + kr, store_mask, a_reg[13] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 14 ) * KC ) + kr, store_mask, a_reg[14] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 15 ) * KC ) + kr, store_mask, a_reg[15] );
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

            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr , a_reg[2] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr , a_reg[3] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr , a_reg[4] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr , a_reg[5] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr , a_reg[6] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr , a_reg[7] );
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

            get_store_mask(&store_mask, k_left);

            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, store_mask, a_reg[1] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr, store_mask , a_reg[2] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr, store_mask , a_reg[3] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr, store_mask , a_reg[4] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr, store_mask , a_reg[5] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr, store_mask , a_reg[6] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr, store_mask , a_reg[7] );
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

            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr , a_reg[2] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr , a_reg[3] );
        }

        if( k_left > 0 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],k_left,( ic + 0 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[1],k_left,( ic + 1 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[2],k_left,( ic + 2 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[3],k_left,( ic + 3 ), kr);

            get_store_mask(&store_mask, k_left);

            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, store_mask, a_reg[1] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr, store_mask , a_reg[2] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr, store_mask , a_reg[3] );
        }
    }
    for( ; ( ic + 2 - 1 ) < MC; ic += 2 )
    {
        for( kr = 0; ( kr + 8 - 1 ) < KC; kr += 8 )
        {
            LOAD_AND_CONVERT_BF16_F32(a_reg[0], ( ic + 0 ) );
            LOAD_AND_CONVERT_BF16_F32(a_reg[1], ( ic + 1 ) );

            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
        }

        if( k_left > 0 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],k_left,( ic + 0 ), kr);
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[1],k_left,( ic + 1 ), kr);

            get_store_mask(&store_mask, k_left);

            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, store_mask, a_reg[0] );
            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, store_mask, a_reg[1] );
        }
    }
    for( ; ( ic ) < MC; ic += 1 )
    {
        for( kr = 0; ( kr + 8 - 1 ) < KC; kr += 8 )
        {
            LOAD_AND_CONVERT_BF16_F32(a_reg[0], ( ic + 0 ) );

            _mm256_storeu_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
        }
        for( ; ( kr + 4 - 1 ) < KC; kr += 4 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],4,( ic + 0 ), kr);

            get_store_mask(&store_mask, 4);

            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , store_mask, a_reg[0] );
        }
        for( ; ( kr + 2 - 1 ) < KC; kr += 2 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],2,( ic + 0 ), kr);

            get_store_mask(&store_mask, 2);

            _mm256_maskstore_ps ( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , store_mask, a_reg[0] );
        }
        for( ; ( kr ) < KC; kr += 1 )
        {
            CVT_BF16_F32_SHIFT_AVX2_lt8(a_reg[0],1,( ic + 0 ), kr);

            get_store_mask(&store_mask, 1);

            _mm256_maskstore_ps( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , store_mask, a_reg[0] );
        }
    }
    *rs_p = KC;
    *cs_p = 1;
}

void cvt_pack_bf16_f32(
      float*	      pack_a_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
    )
{
    if( cs_a == 1 )
    {
        cvt_pack_bf16_f32_row_major( pack_a_buffer, a, rs_a, cs_a, MC, KC, rs_p, cs_p );
    }
    else
    {
        /*ToDo: WIP*/
        //cvt_pack_bf16_f32_col_major( pack_a_buffer, a, rs_a, cs_a, MC, KC );
    }
}
#endif